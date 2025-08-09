import sys
import os
import pickle
import numpy as np

# --- 1. 导入您的模块 (已按您的要求修正) ---
sys.path.append("../")

# a. 从 core.ocp 模块只导入 OCP 函数
from core.ocp import get_graphite_ocp, get_nmc_ocp

# b. 从 core.physics_model 导入 SPM_Thermal 类和两个熵热函数
from core.physics_model import (
    SPM_Thermal,
    nmc_entropic_change,
    graphite_entropic_change
)

# --- 2. 主程序入口 ---
if __name__ == "__main__":

    # a. 设置仿真参数
    Dps = (4e-15,)
    Dns = (3e-14,)
    current = -2.36  # 单位: A/m^2

    # b. 定义模型所需的常量 (最大浓度)
    Cp_max_val = 63104.0
    Cn_max_val = 33133.0


    # c. 创建熵热函数的包装器
    def dUp_dT_wrapper(c):
        sto = c / Cp_max_val
        return nmc_entropic_change(sto)


    def dUn_dT_wrapper(c):
        sto = c / Cn_max_val
        return graphite_entropic_change(sto)


    # d. 循环并运行仿真
    for n, (Dp, Dn) in enumerate(zip(Dps, Dns)):

        # e. 初始化 SPM_Thermal 模型
        model = SPM_Thermal(
            # 电化学参数
            Up=get_nmc_ocp,
            Cp_0=17038, Cp_max=Cp_max_val, Rp=5.22e-6, ep_s=0.665, Lp=75.6e-6, kp=3.6e-11, Dp=Dp,
            Un=get_graphite_ocp,
            Cn_0=29866, Cn_max=Cn_max_val, Rn=5.86e-6, en_s=0.75, Ln=85.2e-6, kn=9e-11, Dn=Dn,
            Ce=1300, R_cell=3.24e-4,

            # 热模型参数 (这些是示例值，请根据您的电芯进行修改)
            m_cell=0.1,  # 电芯质量 [kg]
            cp_cell=1000,  # 电芯平均比热容 [J/(kg.K)]
            A_cell=0.05,  # 电芯散热表面积 [m^2]
            h_conv=0.5,  # 对流换热系数 [W/(m^2.K)]
            T_amb=298.15  # 环境温度 [K]
        )

        # f. 运行模拟
        print(f"正在运行 SPM-热耦合模型仿真: Dp={Dp:.1e}, Dn={Dn:.1e}, Current={current} A/m^2")
        duration = 77800
        delta_t = 10
        print(f"--- 调试信息 ---")
        print(f"即将调用 solve 方法，参数为: duration = {duration}, delta_t = {delta_t}")
        data = model.solve(duration=duration, current_density=current, delta_t=delta_t)

        # g. 检查 solve 方法的输出结果
        print("solve 方法已返回。")
        if data and data.get(model.time_col):
            print(f"返回的数据点数量为: {len(data[model.time_col])}")
        else:
            print("!!! 警告：solve 方法返回了空的数据！仿真未执行任何步骤。!!!")

        # h. 保存模拟结果
        output_filename = f"spm_thermal_{n}_I={current:.3f}_Dp={Dp:.1e}_Dn={Dn:.1e}.pkl"
        output_dir = "../data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_filename)

        print(f"正在将结果保存至: {output_path}")
        with open(output_path, "wb") as binary_file:
            pickle.dump(data, binary_file)
        print("保存成功！\n")