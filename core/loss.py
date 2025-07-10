import torch
from torch import Tensor, mean, nn, square


class PINNLoss(nn.Module):
    def __init__(self, weights: dict, **kwargs):
        super().__init__()
        self.w_data = weights.get("data", 1.0)
        self.w_ic = weights.get("ic", 1.0)
        self.w_bc_surf = weights.get("bc_surf", 1.0)
        self.w_bc_center = weights.get("bc_center", 1.0)
        self.w_pde = weights.get("pde", 1.0)

        self.R_norm = kwargs.get("R_norm")
        self.T_norm = kwargs.get("T_norm")
        if self.R_norm is None or self.T_norm is None:
            raise ValueError("R_norm and T_norm must be provided for loss calculation.")

    def forward(
            self, V_pred: Tensor, V_true: Tensor, Xp: Tensor, Xn: Tensor, model
    ) -> Tensor:
        # 1. 提取变量
        Cp, Cp0_true, Cp_max_param = model.Cp, model.Cp_0, model.Cp_max
        Cn, Cn0_true, Cn_max_param = model.Cn, model.Cn_0, model.Cn_max
        jp, jn = model.jp, model.jn
        Dp, Dn = model.Dp, model.Dn
        device = V_pred.device

        # 2. 数据吻合损失
        loss_data = mean(square(V_pred - V_true))

        # 启用梯度跟踪
        Cp.requires_grad_()
        Xp.requires_grad_()
        Cn.requires_grad_()
        Xn.requires_grad_()

        # --- 3. 动态、稳健地找到边界和初始点 ---
        t_initial = torch.min(Xp[:, 0])
        r_center = torch.min(Xp[:, 1])
        r_surface = torch.max(Xp[:, 1])

        ic_mask_p = torch.isclose(Xp[:, 0], t_initial)
        ic_mask_n = torch.isclose(Xn[:, 0], t_initial)
        center_mask_p = torch.isclose(Xp[:, 1], r_center)
        center_mask_n = torch.isclose(Xn[:, 1], r_center)
        surface_mask_p = torch.isclose(Xp[:, 1], r_surface)
        surface_mask_n = torch.isclose(Xn[:, 1], r_surface)

        # --- 4. 计算各项物理损失 ---

        # 4a. 初始条件损失 (IC)
        Cp_max_tensor = torch.tensor(Cp_max_param, device=device, dtype=torch.float32)
        Cn_max_tensor = torch.tensor(Cn_max_param, device=device, dtype=torch.float32)
        loss_ic_p_raw = mean(square(Cp[ic_mask_p] - Cp0_true))
        loss_ic_n_raw = mean(square(Cn[ic_mask_n] - Cn0_true))
        loss_ic = (loss_ic_p_raw / square(Cp_max_tensor)) + (loss_ic_n_raw / square(Cn_max_tensor))

        # 计算一阶梯度
        Cp_grad = torch.autograd.grad(Cp, Xp, grad_outputs=torch.ones_like(Cp), create_graph=True)[0]
        Cn_grad = torch.autograd.grad(Cn, Xn, grad_outputs=torch.ones_like(Cn), create_graph=True)[0]
        dCp_dt_norm, dCp_dr_norm = Cp_grad[:, 0], Cp_grad[:, 1]
        dCn_dt_norm, dCn_dr_norm = Cn_grad[:, 0], Cn_grad[:, 1]

        # 4b. 中心边界条件损失 (BC_Center)
        loss_bc_center = mean(square(dCp_dr_norm[center_mask_p])) + mean(square(dCn_dr_norm[center_mask_n]))

        # 4c. 表面边界条件损失 (BC_Surf)
        loss_bc_surf_p = mean(square(dCp_dr_norm[surface_mask_p] + (jp.squeeze() * self.R_norm / (Dp + 1e-12))))
        loss_bc_surf_n = mean(square(dCn_dr_norm[surface_mask_n] + (jn.squeeze() * self.R_norm / (Dn + 1e-12))))
        loss_bc_surf = loss_bc_surf_p + loss_bc_surf_n

        # 4d. PDE方程损失
        d2Cp_dr2_norm = \
        torch.autograd.grad(dCp_dr_norm, Xp, grad_outputs=torch.ones_like(dCp_dr_norm), create_graph=True)[0][:, 1]
        d2Cn_dr2_norm = \
        torch.autograd.grad(dCn_dr_norm, Xn, grad_outputs=torch.ones_like(dCn_dr_norm), create_graph=True)[0][:, 1]

        # 提取归一化半径 ρ
        rho_p = Xp[:, 1]
        rho_n = Xn[:, 1]

        # 计算归一化PDE的右侧项
        pde_const_p = Dp * self.T_norm / (self.R_norm ** 2)
        pde_const_n = Dn * self.T_norm / (self.R_norm ** 2)

        # 排除中心点 r=0 (ρ=-1)
        pde_mask_p = ~center_mask_p
        pde_mask_n = ~center_mask_n

        pde_rhs_p = pde_const_p * (d2Cp_dr2_norm[pde_mask_p] + (2 / rho_p[pde_mask_p]) * dCp_dr_norm[pde_mask_p])
        pde_rhs_n = pde_const_n * (d2Cn_dr2_norm[pde_mask_n] + (2 / rho_n[pde_mask_n]) * dCn_dr_norm[pde_mask_n])

        # 计算PDE残差，并用最大浓度的平方进行归一化
        pde_residual_p = dCp_dt_norm[pde_mask_p] - pde_rhs_p
        pde_residual_n = dCn_dt_norm[pde_mask_n] - pde_rhs_n
        loss_pde = mean(square(pde_residual_p / Cp_max_param)) + mean(square(pde_residual_n / Cn_max_param))

        # 5. 加权求和
        total_loss = (self.w_data * loss_data + self.w_ic * loss_ic +
                      self.w_bc_center * loss_bc_center + self.w_bc_surf * loss_bc_surf +
                      self.w_pde * loss_pde)

        # (打印语句已在之前版本中更新，此处不再重复)
        print(
            f"Data:{(self.w_data * loss_data).item():.2E} | "
            f"IC:{(self.w_ic * loss_ic).item():.2E} | "
            f"BC_Surf:{(self.w_bc_surf * loss_bc_surf).item():.2E} | "
            f"BC_Center:{(self.w_bc_center * loss_bc_center).item():.2E} | "
            f"PDE:{(self.w_pde * loss_pde).item():.2E}"
        )

        return total_loss