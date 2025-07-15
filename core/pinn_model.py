# core/pinn_model.py

from typing import Callable
import torch
from torch import Tensor, nn, sqrt, asinh, exp

# DNN类的定义保持不变
class DNN(nn.Module):
    # ... （代码与之前相同，此处省略）
    def __init__(
            self, input_size: int, hidden_size: int, output_size: int, num_hidden_layers: int
    ):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Tanh(),
            *[nn.Linear(hidden_size, hidden_size), nn.Tanh()] * num_hidden_layers,
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.dnn(X)

# SPM_PINN类的定义需要修改
class SPM_PINN(nn.Module):
    # ... (__init__ 和 arrhenius 方法保持不变) ...
    def __init__(self, **kwargs):
        super().__init__()
        # ... (所有初始化代码保持不变) ...
        self.Up, self.Un = kwargs.get("Up"), kwargs.get("Un")
        self.Cp_0, self.Cp_max, self.Rp = kwargs.get("Cp_0"), kwargs.get("Cp_max"), kwargs.get("Rp")
        self.ep_s, self.Lp, self.kp_ref, self.Dp_ref = kwargs.get("ep_s"), kwargs.get("Lp"), kwargs.get(
            "kp"), kwargs.get("Dp")
        self.ap = 3 * self.ep_s / self.Rp
        self.Cn_0, self.Cn_max, self.Rn = kwargs.get("Cn_0"), kwargs.get("Cn_max"), kwargs.get("Rn")
        self.en_s, self.Ln, self.kn_ref, self.Dn_ref = kwargs.get("en_s"), kwargs.get("Ln"), kwargs.get(
            "kn"), kwargs.get("Dn")
        self.an = 3 * self.en_s / self.Rn
        self.Ce, self.R_cell = kwargs.get("Ce"), kwargs.get("R_cell")
        self.F, self.R = 96485.33, 8.314
        self.T_ref, self.T_amb = kwargs.get("T_ref"), kwargs.get("T_amb")
        self.E_D, self.E_k = kwargs.get("E_D"), kwargs.get("E_k")
        self.h_conv = kwargs.get("h_conv")
        self.A_surf = kwargs.get("A_surf")
        self.duration = kwargs.get("duration")
        self.T_dnn = DNN(input_size=1, hidden_size=32, output_size=1, num_hidden_layers=4)
        self.Cp_dnn = DNN(2, kwargs.get("nn_hidden_size"), 1, kwargs.get("nn_num_hidden_layers"))
        self.Cn_dnn = DNN(2, kwargs.get("nn_hidden_size"), 1, kwargs.get("nn_num_hidden_layers"))

    def arrhenius(self, k_ref: Tensor, E_a: float, T: Tensor) -> Tensor:
        return k_ref * exp(E_a / self.R * (1 / self.T_ref - 1 / T))

    def forward(self, I: Tensor, Xp: Tensor, Xn: Tensor, N_t: int) -> Tensor:
        t_norm = Xp[:, 0:1]
        t_unnorm = (t_norm + 1) / 2 * self.duration
        T_nn_output = self.T_dnn(t_norm)
        T_phys = self.T_amb + t_unnorm * T_nn_output
        self.T = T_phys

        Dp_T = self.arrhenius(torch.tensor(self.Dp_ref, device=T_phys.device), self.E_D, T_phys)
        kp_T = self.arrhenius(torch.tensor(self.kp_ref, device=T_phys.device), self.E_k, T_phys)
        Dn_T = self.arrhenius(torch.tensor(self.Dn_ref, device=T_phys.device), self.E_D, T_phys)
        kn_T = self.arrhenius(torch.tensor(self.kn_ref, device=T_phys.device), self.E_k, T_phys)
        self.Dp_T, self.Dn_T = Dp_T, Dn_T

        Cp = self.Cp_0 + t_unnorm * self.Cp_dnn(Xp)
        Cn = self.Cn_0 + t_unnorm * self.Cn_dnn(Xn)
        self.Cp, self.Cn = Cp, Cn

        epsilon = 1e-9
        Cp_clamped = torch.clamp(Cp, min=epsilon, max=self.Cp_max - epsilon)
        Cn_clamped = torch.clamp(Cn, min=epsilon, max=self.Cn_max - epsilon)
        self.Cp_clamped, self.Cn_clamped = Cp_clamped, Cn_clamped
        Cp_surf = Cp_clamped[-N_t:]
        Cn_surf = Cn_clamped[-N_t:]
        T_surf = T_phys[-N_t:]  # <-- 我们已经在这里提取了表面温度

        U_ocv = self.Up(Cp_surf) - self.Un(Cn_surf)
        mp_denominator = (self.F * kp_T[-N_t:] * self.Lp * self.ap * sqrt(self.Cp_max - Cp_surf) * sqrt(
            Cp_surf) * self.Ce ** 0.5)
        mn_denominator = (self.F * kn_T[-N_t:] * self.Ln * self.an * sqrt(self.Cn_max - Cn_surf) * sqrt(
            Cn_surf) * self.Ce ** 0.5)
        mp = I / (mp_denominator + epsilon)
        mn = I / (mn_denominator + epsilon)

        kinetics_const = 2 * self.R * T_surf
        eta_p = (kinetics_const / self.F) * asinh(mp / 2)
        eta_n = (kinetics_const / self.F) * asinh(mn / 2)
        V = U_ocv + eta_p - eta_n + I * self.R_cell

        self.Q_gen = I * (V - U_ocv)

        # --- 【核心修正】: Q_loss 的计算只使用表面温度 T_surf ---
        # 这样 Q_loss 的尺寸就和 Q_gen 一致了 (都为 1500)
        self.Q_loss = self.h_conv * self.A_surf * (T_surf - self.T_amb)

        self.jp = I / (self.F * self.ap * self.Lp)
        self.jn = -I / (self.F * self.an * self.Ln)

        return V