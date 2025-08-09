# -*- coding: utf-8 -*-
"""
一个将单颗粒模型(SPM)与集总参数热模型耦合的Python建模文件。
此文件仅包含模型定义，不包含运行脚本。
"""
from typing import Callable
import numpy as np
from fipy import CellVariable, DiffusionTerm, ExplicitDiffusionTerm, SphericalGrid1D, TransientTerm
from numpy import exp, tanh, arctan, sqrt, log, ndarray


# --- 1. 定义熵热函数 (来自您提供的数据集) ---
# 这部分是模型的依赖，需要放在模型定义的前面或同一个模块内

def nmc_entropic_change(sto: float) -> float:
    """NMC正极的熵热系数 dU/dT (单位: V/K)"""
    a1, a2, b1, b2, c1, c2 = 0.04006, -0.06656, 0.2828, 0.8032, 0.0009855, 0.02179
    # 原始数据单位为 mV/K，除以1000转换为 V/K
    dUdT = (a1 * exp(-((sto - b1) ** 2) / c1) + a2 * exp(-((sto - b2) ** 2) / c2)) / 1000
    return dUdT


def graphite_entropic_change(sto: float) -> float:
    """石墨负极的熵热系数 dU/dT (单位: V/K)"""
    a0, a1, a2 = -0.1112, 0.0, 0.3561
    b1, b2, c0, c1, c2, d1 = 0.4955, 0.08309, 0.02914, 0.1122, 0.004616, 63.9
    # 原始数据单位为 mV/K，除以1000转换为 V/K
    dUdT = (
                   a0 * sto + c0 + a2 * exp(-((sto - b2) ** 2) / c2)
                   + a1 * (tanh(d1 * (sto - (b1 - c1))) - tanh(d1 * (sto - (b1 + c1))))
           ) / 1000
    return dUdT


# --- 2. 定义核心模型类 ---

class SolidDiffusionSolver:
    """
    使用 FiPy 求解球形坐标系下的一维扩散方程。
    此类与您的原始代码结构相同，但增加了一个更适合循环调用的 solve_step 方法。
    """

    def __init__(
            self,
            c0: float,
            D: float,
            R: float,
            nr: int = 20,
            alpha: float = 0.5
    ):
        self.D = D
        self.mesh = SphericalGrid1D(nr=nr, Lr=R)
        self.conc = CellVariable(mesh=self.mesh, name="c", value=c0)
        self.conc.faceGrad.constrain([0.0], self.mesh.facesLeft)
        self.equation = TransientTerm() == DiffusionTerm(coeff=alpha * D) + ExplicitDiffusionTerm(
            coeff=(1.0 - alpha) * D)

    def solve_step(self, dt: float, j: float):
        """为耦合模型设计的、仅求解一个时间步长的方法"""
        if dt > 0:
            mass_flux_bc = -j / self.D
            self.conc.faceGrad.constrain([mass_flux_bc], self.mesh.facesRight)
            self.equation.solve(var=self.conc, dt=dt)


class SPM_Thermal:
    """
    在您的SPM基础上修改的、与集总参数热模型耦合的模型。
    """
    # 定义输出数据字典的键名
    time_col = "Time [s]"
    current_col = "Current [A/m2]"
    voltage_col = "Voltage [V]"
    temp_col = "Temperature [K]"
    q_irr_col = "不可逆热 [W/m2]"
    q_rev_col = "可逆热 [W/m2]"
    ocp_col = "OCP [V]"
    cp_surf_col = "正极表面浓度 [mol/m3]"
    cn_surf_col = "负极表面浓度 [mol/m3]"

    # 您也可以按需添加更多输出列，如 cp_col, cn_col 等

    def __init__(
            self,
            # 您原有的电化学参数
            Up: Callable, Un: Callable,
            Cp_0: float, Cp_max: float, Rp: float, ep_s: float, Lp: float, kp: float, Dp: float,
            Cn_0: float, Cn_max: float, Rn: float, en_s: float, Ln: float, kn: float, Dn: float,
            Ce: float, R_cell: float,
            # 新增的热模型参数
            m_cell: float, cp_cell: float, A_cell: float, h_conv: float, T_amb: float
    ):
        # 将所有传入的参数保存为类的属性
        # 电化学参数
        self.Up, self.Un = Up, Un
        self.Cp_0, self.Cp_max, self.Rp, self.ep_s, self.Lp, self.kp, self.Dp = Cp_0, Cp_max, Rp, ep_s, Lp, kp, Dp
        self.Cn_0, self.Cn_max, self.Rn, self.en_s, self.Ln, self.kn, self.Dn = Cn_0, Cn_max, Rn, en_s, Ln, kn, Dn
        self.Ce, self.R_cell = Ce, R_cell
        # 热力学参数
        self.m_cell, self.cp_cell, self.A_cell, self.h_conv, self.T_amb = m_cell, cp_cell, A_cell, h_conv, T_amb

        # 根据输入参数计算派生参数
        self.ap = 3 * self.ep_s / self.Rp
        self.an = 3 * self.en_s / self.Rn

        # 通用常数
        self.F, self.R = 96485.33, 8.314

        # 初始化模型的状态变量
        self.T_cell = self.T_amb  # 电芯初始温度等于环境温度
        self.Cp_solver = SolidDiffusionSolver(c0=self.Cp_0, D=self.Dp, R=self.Rp)
        self.Cn_solver = SolidDiffusionSolver(c0=self.Cn_0, D=self.Dn, R=self.Rn)

    def solve(self, duration: float, current_density: float, delta_t: float = 1.0) -> dict:
        time_steps = np.arange(0, duration + delta_t, delta_t)
        jp = current_density / (self.F * self.ap * self.Lp)
        jn = -current_density / (self.F * self.an * self.Ln)

        data = {k: [] for k in [self.time_col, self.current_col, self.voltage_col, self.temp_col,
                                self.q_irr_col, self.q_rev_col, self.ocp_col,
                                self.cp_surf_col, self.cn_surf_col]}

        for t in time_steps:
            cp_surf = self.Cp_solver.conc.value[-1]
            cn_surf = self.Cn_solver.conc.value[-1]
            U_ocp = self.Up(cp_surf) - self.Un(cn_surf)

            try:
                cp_surf_clipped = np.clip(cp_surf, 1e-9, self.Cp_max - 1e-9)
                cn_surf_clipped = np.clip(cn_surf, 1e-9, self.Cn_max - 1e-9)
                mp = current_density / (self.F * self.kp * self.Lp * self.ap * sqrt(
                    cp_surf_clipped * (self.Cp_max - cp_surf_clipped) * self.Ce))
                mn = current_density / (self.F * self.kn * self.Ln * self.an * sqrt(
                    cn_surf_clipped * (self.Cn_max - cn_surf_clipped) * self.Ce))
                kinetics_const = 2 * self.R * self.T_cell / self.F
                eta_p = kinetics_const * log((sqrt(mp ** 2 + 4) + mp) / 2)
                eta_n = kinetics_const * log((sqrt(mn ** 2 + 4) + mn) / 2)
                V_terminal = U_ocp + eta_p + eta_n + current_density * self.R_cell
            except ValueError:
                print(f"警告: 在 t={t:.1f}s 时发生数学计算错误, 仿真提前终止。")
                break

            q_irr = current_density * (V_terminal - U_ocp)
            sto_p = cp_surf / self.Cp_max
            sto_n = cn_surf / self.Cn_max
            dU_dT = nmc_entropic_change(sto_p) - graphite_entropic_change(sto_n)
            q_rev = current_density * self.T_cell * dU_dT
            q_gen_flux = q_irr + q_rev

            # --- 关键修复点：使用更合理的总功率计算 ---
            # 对于集总模型，我们假设产热功率正比于电芯的散热面积 A_cell
            # 这将产热通量 [W/m^2] 转换为了总功率 [W]
            Q_gen = q_gen_flux * self.A_cell
            # --------------------------------------------

            Q_conv = self.h_conv * self.A_cell * (self.T_cell - self.T_amb)
            self.T_cell += (Q_gen - Q_conv) / (self.m_cell * self.cp_cell) * delta_t

            for col, val in zip(data.keys(),
                                [t, current_density, V_terminal, self.T_cell, q_irr, q_rev, U_ocp, cp_surf, cn_surf]):
                data[col].append(val)

            self.Cp_solver.solve_step(dt=delta_t, j=jp)
            self.Cn_solver.solve_step(dt=delta_t, j=jn)

        return data