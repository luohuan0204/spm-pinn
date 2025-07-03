import sys

sys.path.append("../")
import pickle

from core.ocp import get_graphite_ocp, get_nmc_ocp
from core.physics_model import SPM

if __name__ == "__main__":
    #设置正负极扩散系数
    Dps = (1e-14,)
    Dns = (3e-14,)

    for n, (Dp, Dn) in enumerate(zip(Dps, Dns)):
        #初始化spm模型
        spm = SPM(
            Up=get_nmc_ocp,  # Positive electrode OCP as f(conc) [V]
            #通过控制初始的正负极浓度可以设置电池的起始电压
            Cp_0=17038,  # Initial positive electrode Li concentration [mol/m3]
            Cp_max=63104,  # Max positive electrode Li concentration [mol/m3]
            Rp=5.22e-6,  # Positive electrode particle radius [m]
            ep_s=0.335,  # Positive electrode volume fraction [-]
            Lp=75.6e-6,  # Positive Electrode thickness [m]
            kp=5e-10,  # Positive electrode reaction rate constant [m^2.5/(mol^0.5.s)]
            Dp=Dp,  # Positive electrode diffusivity [m2/s]
            Un=get_graphite_ocp,  # Negative electrode OCP as f(conc) [V]
            #初始的负极浓度
            Cn_0=29866,  # Initial negative electrode Li concentration [mol/m3]
            Cn_max=33133,  # Max negative electrode Li concentration [mol/m3]
            Rn=5.22e-6,  # Negative electrode particle radius [m]
            en_s=0.75,  # Negative electrode volume fraction [-]
            Ln=75.6e-6,  # Negative Electrode thickness [m]
            kn=5e-10,  # Negative electrode reaction rate constant [m^2.5/(mol^0.5.s)]
            Dn=Dn,  # Negative electrode diffusivity [m2/s]
            Ce=1000,  # Electrolyte Li concentration [mol/m3]
            R_cell=3.24e-4,  # Cell resistance [ohm m2]
        )
        # 运行模拟
        data = spm.solve(duration=1000, current_density=-20, delta_t=10)
        #保存模拟结果
        with open(f"../data/spm{n}_Dp={Dp}_Dn={Dn}", "ab") as binary_file:
            pickle.dump(data, binary_file)
