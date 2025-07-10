from typing import Callable

import torch
from torch import Tensor, nn, sqrt, asinh


# DNN类保持不变
class DNN(nn.Module):
    def __init__(
            self, input_size: int, hidden_size: int, output_size: int, num_hidden_layers: int
    ):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Tanh(),
            *[nn.Linear(hidden_size, hidden_size), nn.Tanh()] * num_hidden_layers,
            nn.Linear(hidden_size, output_size), nn.Tanh(),
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.dnn(X)


# SPM_PINN类是我们添加打印语句的地方
class SPM_PINN(nn.Module):
    def __init__(self, **kwargs):  # 使用kwargs简化，保持不变
        super().__init__()
        # 为了简洁，这里省略了所有参数的赋值，假设它们和您原来的代码一样
        self.nn_hidden_size = kwargs.get("nn_hidden_size")
        self.nn_num_hidden_layers = kwargs.get("nn_num_hidden_layers")
        self.Up, self.Un = kwargs.get("Up"), kwargs.get("Un")
        self.Cp_0, self.Cp_max, self.Rp = kwargs.get("Cp_0"), kwargs.get("Cp_max"), kwargs.get("Rp")
        self.ep_s, self.Lp, self.kp, self.Dp = kwargs.get("ep_s"), kwargs.get("Lp"), kwargs.get("kp"), kwargs.get("Dp")
        self.ap = 3 * self.ep_s / self.Rp
        self.Cn_0, self.Cn_max, self.Rn = kwargs.get("Cn_0"), kwargs.get("Cn_max"), kwargs.get("Rn")
        self.en_s, self.Ln, self.kn, self.Dn = kwargs.get("en_s"), kwargs.get("Ln"), kwargs.get("kn"), kwargs.get("Dn")
        self.an = 3 * self.en_s / self.Rn
        self.Ce, self.R_cell = kwargs.get("Ce"), kwargs.get("R_cell")
        self.T, self.F, self.R = 298, 96485.33, 8.314
        self.Cp_dnn = DNN(2, self.nn_hidden_size, 1, self.nn_num_hidden_layers)
        self.Cn_dnn = DNN(2, self.nn_hidden_size, 1, self.nn_num_hidden_layers)

    def unnormalize_data(self, C_norm: Tensor, max_value: float) -> Tensor:
        return (C_norm + 1) / 2 * max_value

    def forward(self, I: Tensor, Xp: Tensor, Xn: Tensor, N_t: int) -> Tensor:
        print("\n--- Entering forward pass ---")

        # 1. 神经网络预测
        Cp_norm = self.Cp_dnn(Xp)
        Cn_norm = self.Cn_dnn(Xn)
        print(f"Step 1: NN output Cp_norm min/max: {Cp_norm.min().item():.2E}, {Cp_norm.max().item():.2E}")

        # 2. 反归一化
        Cp = self.unnormalize_data(Cp_norm, self.Cp_max)
        Cn = self.unnormalize_data(Cn_norm, self.Cn_max)
        print(f"Step 2: Unnormalized Cp min/max: {Cp.min().item():.2E}, {Cp.max().item():.2E}")

        # 3. 裁剪
        epsilon = 1e-9
        Cp_clamped = torch.clamp(Cp, min=epsilon, max=self.Cp_max - epsilon)
        Cn_clamped = torch.clamp(Cn, min=epsilon, max=self.Cn_max - epsilon)
        print(f"Step 3: Clamped Cp min/max: {Cp_clamped.min().item():.2E}, {Cp_clamped.max().item():.2E}")

        # 4. 提取表面浓度
        Cp_surf = Cp_clamped[-N_t:]
        Cn_surf = Cn_clamped[-N_t:]
        print(f"Step 4: Surface Cp_surf min/max: {Cp_surf.min().item():.2E}, {Cp_surf.max().item():.2E}")

        # 5. 计算分母中的开方项
        sqrt_term1_p = sqrt(self.Cp_max - Cp_surf)
        sqrt_term2_p = sqrt(Cp_surf)
        print(f"Step 5.1: sqrt(Cp_max - Cp_surf) contains NaN: {torch.isnan(sqrt_term1_p).any().item()}")
        print(f"Step 5.2: sqrt(Cp_surf) contains NaN: {torch.isnan(sqrt_term2_p).any().item()}")

        # 6. 计算分母
        mp_denominator = (
                self.F * self.kp * self.Lp * self.ap
                * sqrt_term1_p * sqrt_term2_p * self.Ce ** 0.5
        )
        print(f"Step 6: mp_denominator min/max: {mp_denominator.min().item():.2E}, {mp_denominator.max().item():.2E}")

        # 7. 计算mp
        mp = I / (mp_denominator + epsilon)
        print(f"Step 7: mp contains NaN: {torch.isnan(mp).any().item()}, contains inf: {torch.isinf(mp).any().item()}")

        # 为了简洁，我们暂时只详细调试正极部分
        mn_denominator = (
                self.F * self.kn * self.Ln * self.an
                * sqrt(self.Cn_max - Cn_surf) * sqrt(Cn_surf) * self.Ce ** 0.5
        )
        mn = I / (mn_denominator + epsilon)

        # 8. 计算电压
        kinetics_const = 2 * self.R * self.T / self.F
        Up_val = self.Up(Cp_surf)
        Un_val = self.Un(Cn_surf)
        asinh_mp = asinh(mp / 2)
        asinh_mn = asinh(mn / 2)
        ohmic_val = I * self.R_cell

        print(f"Step 8.1: OCP Up contains NaN: {torch.isnan(Up_val).any().item()}")
        print(
            f"Step 8.2: asinh_mp contains NaN: {torch.isnan(asinh_mp).any().item()}, contains inf: {torch.isinf(asinh_mp).any().item()}")

        V = Up_val - Un_val + kinetics_const * asinh_mp + kinetics_const * asinh_mn + ohmic_val
        print(f"Step 8.3: Final Voltage V contains NaN: {torch.isnan(V).any().item()}")

        # 9. 保存变量
        self.Cp, self.Cn = Cp, Cn

        self.Cp_clamped = Cp_clamped
        self.Cn_clamped = Cn_clamped

        self.jp = I / (self.F * self.ap * self.Lp)
        self.jn = -I / (self.F * self.an * self.Ln)

        print("--- Exiting forward pass ---\n")
        return V