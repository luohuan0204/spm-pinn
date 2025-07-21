import sys
import os
import pickle

# 假设您的核心代码在上一级目录的core文件夹中
# 如果目录结构不同，请相应调整
sys.path.append("../")
from core.ocp import get_graphite_ocp, get_nmc_ocp
from core.physics_model import SPM

if __name__ == "__main__":
    # --- 1. 设置仿真参数 ---
    # 设置要测试的扩散系数对
    Dps = (4e-15,)
    Dns = (3e-14,)

    # 将电流密度定义为一个变量，以区分充放电
    # 负值代表放电，正值代表充电
    current = -2.36  # 单位: A/m^2

    # --- 2. 循环并运行仿真 ---
    for n, (Dp, Dn) in enumerate(zip(Dps, Dns)):
        # 初始化spm模型
        spm = SPM(
            Up=get_nmc_ocp,  # Positive electrode OCP as f(conc) [V]
            Cp_0=17038,  # Initial positive electrode Li concentration [mol.m-3]
            Cp_max=63104,  # Max positive electrode Li concentration [mol.m-3]
            Rp=5.22e-6,  # Positive electrode particle radius [m]
            ep_s=0.665,  # Positive electrode volume fraction [-]
            Lp=75.6e-6,  # Positive Electrode thickness [m]
            kp=3.6e-11,  # Positive electrode reaction rate constant [m^2.5/(mol^0.5.s)]
            Dp=Dp,  # Positive electrode diffusivity [m2/s]
            Un=get_graphite_ocp,  # Negative electrode OCP as f(conc) [V]
            Cn_0=29866,  # Initial negative electrode Li concentration [mol.m-3]
            Cn_max=33133,  # Max negative electrode Li concentration [mol.m-3]
            Rn=5.86e-6,  # Negative electrode particle radius [m]
            en_s=0.75,  # Negative electrode volume fraction [-]
            Ln=85.2e-6,  # Negative Electrode thickness [m]
            kn=9e-11,  # Negative electrode reaction rate constant [m^2.5/(mol^0.5.s)]
            Dn=Dn,  # Negative electrode diffusivity [m2/s]
            Ce=1300,  # Electrolyte Li concentration [mol.m-3]
            R_cell=3.24e-4,  # Cell resistance [ohm m2]
        )

        # 运行模拟
        print(f"正在运行仿真: Dp={Dp:.1e}, Dn={Dn:.1e}, Current={current} A/m^2")
        data = spm.solve(duration=77800, current_density=current, delta_t=100)

        # --- 3. 保存模拟结果 (已按要求修改) ---

        # a. 构造包含带符号电流大小的新文件名
        #    - 去掉了 abs() 函数，直接使用 current 变量
        #    - 这样文件名中就会包含负号，例如 I=-49.05
        output_filename = f"spm{n}_I={current:.3f}_Dp={Dp:.1e}_Dn={Dn:.1e}.pkl"

        # b. 确保输出目录存在
        output_dir = "../data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建目录: {output_dir}")

        output_path = os.path.join(output_dir, output_filename)

        # c. 使用 "wb" (write binary) 模式进行重写写入
        print(f"正在将结果保存至: {output_path}")
        with open(output_path, "wb") as binary_file:
            pickle.dump(data, binary_file)

        print("保存成功！\n")