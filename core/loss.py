# core/loss.py

import torch
from torch import Tensor, mean, nn, square


class PINNLoss(nn.Module):
    """
    一个完整的、为电-热耦合SPM模型定制的、数值稳健的PINN损失函数。

    它包含了数据吻合损失 (电压和温度)，以及五项基于物理的损失：
    1. 初始条件 (IC)
    2. 中心边界条件 (BC at r=0)
    3. 表面边界条件 (BC at r=R)
    4. 固相扩散偏微分方程 (PDE)
    5. 热平衡常微分方程 (Thermal ODE)
    """

    def __init__(self, weights: dict, norm_params: dict, thermal_params: dict):
        """
        初始化损失函数。

        Args:
            weights (dict): 一个包含各项损失权重的字典。
            norm_params (dict): 一个包含归一化所需参数的字典。
            thermal_params (dict): 一个包含热模型物理参数的字典。
        """
        super().__init__()
        self.weights = weights
        self.norm_params = norm_params
        self.thermal_params = thermal_params

    def forward(
            self, V_pred: Tensor, V_true: Tensor, T_true: Tensor, Xp: Tensor, Xn: Tensor, model
    ) -> tuple[Tensor, dict]:

        device = V_pred.device
        loss_dict = {
            "data": torch.tensor(0.0, device=device),
            "T_data": torch.tensor(0.0, device=device),  # 新增温度数据损失项
            "ic": torch.tensor(0.0, device=device),
            "bc_center": torch.tensor(0.0, device=device),
            "bc_surf": torch.tensor(0.0, device=device),
            "pde": torch.tensor(0.0, device=device),
            "thermal": torch.tensor(0.0, device=device),
        }

        # ====================================================================
        # 1. 数据吻合损失 (Data Loss)
        # ====================================================================
        # a. 电压数据损失
        if self.weights.get("data", 0.0) > 0:
            loss_dict["data"] = mean(square(V_pred - V_true))

        # b. 新增：温度数据损失
        if self.weights.get("T_data", 0.0) > 0:
            # 从模型中提取预测的温度 (只在表面点，因为T与r无关)
            rR_mask_p = torch.isclose(Xp[:, 1], torch.max(Xp[:, 1]))
            T_pred = model.T[rR_mask_p]
            # 计算温度的均方误差损失
            loss_dict["T_data"] = mean(square(T_pred.squeeze() - T_true.squeeze()))

        # ====================================================================
        # 2. 物理损失 (Physics Loss)
        # ====================================================================
        needs_phys_grads = any(self.weights.get(k, 0.0) > 0 for k in ['ic', 'bc_center', 'bc_surf', 'pde', 'thermal'])
        if needs_phys_grads:
            Xp.requires_grad_(True)
            Xn.requires_grad_(True)

            # --- 2a. 电化学物理损失 ---
            calc_conc_grads = any(self.weights.get(k, 0.0) > 0 for k in ['ic', 'bc_center', 'bc_surf', 'pde'])

            if calc_conc_grads:
                Cp, Cn = model.Cp, model.Cn
                Cp_grad = torch.autograd.grad(Cp.sum(), Xp, create_graph=True)[0]
                dCp_dt_norm, dCp_dr_norm = Cp_grad[:, 0:1], Cp_grad[:, 1:2]
                Cn_grad = torch.autograd.grad(Cn.sum(), Xn, create_graph=True)[0]
                dCn_dt_norm, dCn_dr_norm = Cn_grad[:, 0:1], Cn_grad[:, 1:2]

                t0_mask_p = torch.isclose(Xp[:, 0], torch.min(Xp[:, 0]))
                r0_mask_p = torch.isclose(Xp[:, 1], torch.min(Xp[:, 1]))
                rR_mask_p = torch.isclose(Xp[:, 1], torch.max(Xp[:, 1]))
                t0_mask_n = torch.isclose(Xn[:, 0], torch.min(Xn[:, 0]))
                r0_mask_n = torch.isclose(Xn[:, 1], torch.min(Xn[:, 1]))
                rR_mask_n = torch.isclose(Xn[:, 1], torch.max(Xn[:, 1]))

                if self.weights.get("ic", 0.0) > 0:
                    loss_dict["ic"] = mean(square(Cp[t0_mask_p] - model.Cp_0)) + \
                                      mean(square(Cn[t0_mask_n] - model.Cn_0))

                if self.weights.get("bc_center", 0.0) > 0:
                    loss_dict["bc_center"] = mean(square(dCp_dr_norm[r0_mask_p])) + \
                                             mean(square(dCn_dr_norm[r0_mask_n]))

                if self.weights.get("bc_surf", 0.0) > 0:
                    dCp_dr_phys = dCp_dr_norm[rR_mask_p] * (2 / self.norm_params['R_p'])
                    dCn_dr_phys = dCn_dr_norm[rR_mask_n] * (2 / self.norm_params['R_n'])
                    Dp_surf = model.Dp_T[rR_mask_p]
                    Dn_surf = model.Dn_T[rR_mask_n]
                    residual_bc_p = -Dp_surf * dCp_dr_phys - model.jp
                    residual_bc_n = -Dn_surf * dCn_dr_phys - model.jn
                    loss_dict["bc_surf"] = mean(square(residual_bc_p)) + \
                                           mean(square(residual_bc_n))

                if self.weights.get("pde", 0.0) > 0:
                    term_pde_p = square(Xp[:, 1:2]) * model.Dp_T * dCp_dr_norm
                    d_term_pde_p_dr = torch.autograd.grad(term_pde_p.sum(), Xp, create_graph=True)[0][:, 1:2]
                    term_pde_n = square(Xn[:, 1:2]) * model.Dn_T * dCn_dr_norm
                    d_term_pde_n_dr = torch.autograd.grad(term_pde_n.sum(), Xn, create_graph=True)[0][:, 1:2]
                    dCp_dt_phys = dCp_dt_norm * (2 / self.norm_params['T_max'])
                    dCn_dt_phys = dCn_dt_norm * (2 / self.norm_params['T_max'])
                    laplacian_p = d_term_pde_p_dr * (2 / self.norm_params['R_p']) / (square(Xp[:, 1:2]) + 1e-9)
                    laplacian_n = d_term_pde_n_dr * (2 / self.norm_params['R_n']) / (square(Xn[:, 1:2]) + 1e-9)
                    pde_mask_p = ~r0_mask_p
                    pde_mask_n = ~r0_mask_n
                    residual_pde_p = dCp_dt_phys[pde_mask_p] - laplacian_p[pde_mask_p]
                    residual_pde_n = dCn_dt_phys[pde_mask_n] - laplacian_n[pde_mask_n]
                    loss_dict["pde"] = mean(square(residual_pde_p)) + \
                                       mean(square(residual_pde_n))

            # --- 2b. 热平衡物理损失 ---
            if self.weights.get("thermal", 0.0) > 0:
                T_phys, Q_gen, Q_loss = model.T, model.Q_gen, model.Q_loss
                T_grad = torch.autograd.grad(T_phys.sum(), Xp, create_graph=True, allow_unused=True)[0]
                if T_grad is not None:
                    dT_dt_norm = T_grad[:, 0:1]
                    temp_range = self.norm_params.get('Temp_range', 80.0)
                    dT_dt_phys = dT_dt_norm * (temp_range / self.norm_params['T_max'])
                    rR_mask_p_thermal = torch.isclose(Xp[:, 1], torch.max(Xp[:, 1]))
                    dT_dt_phys_surf = dT_dt_phys[rR_mask_p_thermal]
                    thermal_residual = (
                            self.thermal_params['m_cell'] * self.thermal_params['Cp_cell'] * dT_dt_phys_surf
                            - (Q_gen - Q_loss)
                    )
                    loss_dict["thermal"] = mean(square(thermal_residual / 100.0))

        # ====================================================================
        # 4. 最终加权求和
        # ====================================================================
        total_loss = torch.tensor(0.0, device=device)
        for key, weight in self.weights.items():
            if weight > 0 and key in loss_dict:  # 确保key存在
                total_loss += weight * loss_dict[key]

        return total_loss, loss_dict