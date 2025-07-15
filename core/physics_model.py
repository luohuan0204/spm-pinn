from typing import Callable
import numpy as np

# 为了避免未安装FiPy时报错，我们采用懒加载
try:
    from fipy import CellVariable, SphericalGrid1D, TransientTerm, DiffusionTerm, ExplicitDiffusionTerm

    FIPY_AVAILABLE = True
except ImportError:
    FIPY_AVAILABLE = False


class SolidDiffusionSolver:
    t_col, j_col, r_col = "t [s]", "j [A/m2]", "r [m]"
    c_col, c_surf_col = "c [mol/m3]", "c_surf [mol/m3]"

    def __init__(self, c0: float, D: float, R: float, nr: int = 20):
        if not FIPY_AVAILABLE:
            raise ImportError("FiPy not installed. Please run 'pip install fipy' to generate simulation data.")

        self.D = D
        self.mesh = SphericalGrid1D(nr=nr, Lr=R)
        self.conc = CellVariable(mesh=self.mesh, name=r"$c$", value=c0, hasOld=True)
        self.conc.faceGrad.constrain(0.0, self.mesh.facesLeft)

    def solve_one_step(self, dt: float, j_flux: float):
        self.conc.updateOld()
        alpha = 0.5
        equation = TransientTerm() == DiffusionTerm(coeff=alpha * self.D) + ExplicitDiffusionTerm(
            coeff=(1.0 - alpha) * self.D)
        mass_flux_coeff = -j_flux / (self.D + 1e-30)
        self.conc.faceGrad.constrain(mass_flux_coeff, self.mesh.facesRight)
        equation.solve(var=self.conc, dt=dt)
        return self.conc.value[-1], self.conc.value.copy()


class SPM:
    time_col = "Time [s]"
    current_col = "Current [A/m2]"
    voltage_col = "Voltage [V]"
    rp_col = "Positive Particle Radius [m]"
    cp_col = "Positive Electrode Concentration [mol/m3]"
    cp_surf_col = "Positive Electrode Surface Concentration [mol/m3]"
    rn_col = "Negative Particle Radius [m]"
    cn_col = "Negative Electrode Concentration [mol/m3]"
    cn_surf_col = "Negative Electrode Surface Concentration [mol/m3]"
    temp_col = "Temperature [K]"

    def __init__(
            self,
            Up: Callable, Cp_0: float, Cp_max: float, Rp: float, ep_s: float,
            Lp: float, kp: float, Dp: float,
            Un: Callable, Cn_0: float, Cn_max: float, Rn: float, en_s: float,
            Ln: float, kn: float, Dn: float,
            Ce: float, R_cell: float, I: float,
            m_cell: float, Cp_cell: float, A_surf: float, h_conv: float, T_amb: float,
            T_ref: float, E_D: float, E_k: float,
            **kwargs
    ):
        self.Up, self.Un = Up, Un
        self.Cp_0, self.Cp_max, self.Rp = Cp_0, Cp_max, Rp
        self.ep_s, self.Lp, self.kp_ref, self.Dp_ref = ep_s, Lp, kp, Dp
        self.ap = 3 * self.ep_s / self.Rp
        self.Cn_0, self.Cn_max, self.Rn = Cn_0, Cn_max, Rn
        self.en_s, self.Ln, self.kn_ref, self.Dn_ref = en_s, Ln, kn, Dn
        self.an = 3 * self.en_s / self.Rn
        self.Ce, self.R_cell, self.I = Ce, R_cell, I
        self.R, self.F = 8.314, 96485.33
        self.m_cell, self.Cp_cell = m_cell, Cp_cell
        self.A_surf, self.h_conv = A_surf, h_conv
        self.T_amb, self.T_ref = T_amb, T_ref
        self.E_D, self.E_k = E_D, E_k

    def arrhenius(self, k_ref: float, E_a: float, T: float) -> float:
        return k_ref * np.exp(E_a / self.R * (1 / self.T_ref - 1 / T))

    def solve(self, duration: float, delta_t: float = 1.0) -> dict:
        print("开始运行电-热耦合SPM仿真...")
        sim_time = np.arange(0, duration + delta_t, delta_t)

        T_current = self.T_amb
        cp_solver = SolidDiffusionSolver(c0=self.Cp_0, D=self.Dp_ref, R=self.Rp)
        cn_solver = SolidDiffusionSolver(c0=self.Cn_0, D=self.Dn_ref, R=self.Rn)

        history = {
            self.voltage_col: [], self.cp_surf_col: [], self.cn_surf_col: [],
            self.temp_col: [T_current], self.cp_col: [], self.cn_col: [],
            self.rp_col: [], self.rn_col: []
        }

        for i, t_step in enumerate(sim_time[1:]):
            Dp_T = self.arrhenius(self.Dp_ref, self.E_D, T_current)
            kp_T = self.arrhenius(self.kp_ref, self.E_k, T_current)
            Dn_T = self.arrhenius(self.Dn_ref, self.E_D, T_current)
            kn_T = self.arrhenius(self.kn_ref, self.E_k, T_current)

            cp_solver.D, cn_solver.D = Dp_T, Dn_T

            jp = self.I / (self.F * self.ap * self.Lp)
            jn = -self.I / (self.F * self.an * self.Ln)

            Cp_surf, Cp_profile = cp_solver.solve_one_step(delta_t, jp)
            Cn_surf, Cn_profile = cn_solver.solve_one_step(delta_t, jn)

            epsilon = 1e-9
            Cp_surf_safe = np.clip(Cp_surf, epsilon, self.Cp_max - epsilon)
            Cn_surf_safe = np.clip(Cn_surf, epsilon, self.Cn_max - epsilon)
            U_ocv = self.Up(Cp_surf_safe) - self.Un(Cn_surf_safe)

            mp_denom = (self.F * kp_T * self.Lp * self.ap * np.sqrt(self.Cp_max - Cp_surf_safe) * np.sqrt(
                Cp_surf_safe) * self.Ce ** 0.5)
            mn_denom = (self.F * kn_T * self.Ln * self.an * np.sqrt(self.Cn_max - Cn_surf_safe) * np.sqrt(
                Cn_surf_safe) * self.Ce ** 0.5)
            mp, mn = self.I / (mp_denom + epsilon), self.I / (mn_denom + epsilon)
            kinetics_const = 2 * self.R * T_current
            # 分别计算两个电极的活化过电位
            eta_p = (kinetics_const / self.F) * np.arcsinh(mp / 2)
            eta_n = (kinetics_const / self.F) * np.arcsinh(mn / 2)

            # 应用正确的物理公式：V = U_ocv + eta_p - eta_n + I * R_cell
            V = U_ocv + eta_p - eta_n + self.I * self.R_cell

            Q_irr = self.I * (V - U_ocv)
            Q_loss = self.h_conv * self.A_surf * (T_current - self.T_amb)
            dT = ((Q_irr - Q_loss) / (self.m_cell * self.Cp_cell)) * delta_t
            T_current += dT
            T_current = np.clip(T_current, 273.15, 393.15)

            history[self.voltage_col].append(V)
            history[self.cp_surf_col].append(Cp_surf)
            history[self.cn_surf_col].append(Cn_surf)
            history[self.temp_col].append(T_current)
            history[self.cp_col].append(Cp_profile)
            history[self.cn_col].append(Cn_profile)
            history[self.rp_col].append(cp_solver.mesh.x.value)
            history[self.rn_col].append(cn_solver.mesh.x.value)

            if (i + 1) % 360 == 0:
                print(f"仿真进度: {t_step:.0f}/{duration}s, 当前温度: {T_current - 273.15:.2f}°C")

        # --- 【核心修改】将电流数据也添加到最终的字典中 ---
        history[self.time_col] = sim_time[1:]
        history[self.current_col] = np.full_like(history[self.time_col], self.I)

        print("耦合仿真完成。")
        return history