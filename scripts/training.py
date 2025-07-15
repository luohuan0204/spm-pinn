# scripts/training.py
# 最终、完整、可以直接运行的版本

import sys
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# =============================================================================
# 1. 导入所有需要的模块
# =============================================================================
# 确保项目根目录在Python路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.dataset import SimulationDataset
from core.data_module import DataModule
from core.loss import PINNLoss
from core.ocp_torch import get_graphite_ocp, get_nmc_ocp
from core.pinn_model import SPM_PINN
from core.training_module import TrainingModule
from core.physics_model import SPM


# =============================================================================
# 2. 配置文件 (所有可调参数都集中在这里)
# =============================================================================
def get_config():
    """将所有配置集中管理，方便修改和查阅。"""

    # 动态加载参数，使代码具有通用性
    temp_dataset = SimulationDataset(data_directory="../data_coupled")
    sample_params = temp_dataset.get_params(0)  # 从第一个数据文件中读取参数

    config = {
        "run_name": "spm_pinn_coupled_final_run",  # 为本次运行命名
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        "train_params": {
            "max_epochs": 100,  # 您可以根据需要调整训练的总轮数
            "learning_rate": 5e-4,
        },

        "data_params": {
            "data_directory": "../data_coupled",
            "train_split": 1.0,  # 当前为“仅训练”模式
            "val_split": 0.0,
            "batch_size": 1,
            "seed": 42,
        },

        "loss_params": {
            "weights": {
                "data": 1.0,  # 电压数据损失
                "T_data": 0.0,  # 温度数据损失
                "ic": 1.0,  # 初始条件损失
                "bc_center": 0.0,  # 中心边界条件 (权重调小)
                "bc_surf": 0.0,  # 表面边界条件
                "pde": 1.0,  # PDE方程损失
                "thermal": 0.0,  # 热损失
            },
            "norm_params": {
                "T_max": sample_params['duration'],
                "R_p": sample_params['Rp'],
                "R_n": sample_params['Rn'],
                "Temp_range": 80.0,
            },
            "thermal_params": sample_params
        },

        "model_params": {
            "Up": get_nmc_ocp, "Un": get_graphite_ocp,
            "nn_hidden_size": 128, "nn_num_hidden_layers": 8,
        }
    }
    config["model_params"].update(sample_params)
    return config


# =============================================================================
# 3. 主训练与评估函数
# =============================================================================
def main():
    """主函数，包含完整的训练、加载和绘图流程。"""
    config = get_config()
    pl.seed_everything(config["data_params"]["seed"])

    run_name = config['run_name']
    plots_dir = f"results/{run_name}/plots"
    checkpoints_dir = f"results/{run_name}/checkpoints"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # --- 初始化模块 ---
    print("--- 1. 初始化数据模块 ---")
    dataset = SimulationDataset(data_directory=config["data_params"]["data_directory"])
    datamodule = DataModule(
        dataset=dataset,
        train_split=config["data_params"]["train_split"],
        val_split=config["data_params"]["val_split"],
        batch_size=config["data_params"]["batch_size"],
        seed=config["data_params"]["seed"]
    )
    datamodule.setup(stage='fit')

    print("--- 2. 初始化模型、损失函数和训练模块 ---")
    model = SPM_PINN(**config["model_params"]).to(config["device"])
    loss_fn = PINNLoss(**config["loss_params"])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["train_params"]["learning_rate"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2000)
    training_module = TrainingModule(model, loss_fn, optimizer, lr_scheduler)

    # --- 设置训练器 ---
    logger = TensorBoardLogger("results/logs", name=run_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="best_model-{epoch:02d}-{train/loss_total:.2e}",
        save_top_k=1, verbose=True, monitor="train/loss_total", mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, max_epochs=config["train_params"]["max_epochs"],
        logger=logger, callbacks=[checkpoint_callback, lr_monitor]
    )

    # --- 开始训练 ---
    print("\n--- 3. 开始训练模型 ---")
    trainer.fit(model=training_module, train_dataloaders=datamodule.train_dataloader())
    print("\n--- 训练完成 ---")

    # =============================================================================
    # 4. 结果可视化 (加载最佳模型并绘图)
    # =============================================================================
    print("\n--- 4. 加载最佳模型进行预测和绘图 ---")
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path or not os.path.exists(best_model_path):
        print("警告：未找到最佳模型检查点，将使用训练结束时的模型进行预测。")
        best_model = training_module.model
    else:
        best_model_module = TrainingModule.load_from_checkpoint(
            checkpoint_path=best_model_path, model=model,
            loss_function=loss_fn, optimizer=optimizer
        )
        best_model = best_model_module.model
        print(f"从 {best_model_path} 加载了最佳模型。")

    best_model.eval()
    device = config["device"]
    best_model.to(device)

    # --- 准备预测数据 ---
    I, Xp, Xn, Y_V_true_tensor, Y_T_true_tensor = datamodule.dataset[0]
    I, Xp, Xn = I.to(device), Xp.to(device), Xn.to(device)

    # --- 运行预测 ---
    with torch.no_grad():
        V_pred_tensor = best_model(I, Xp, Xn, Y_V_true_tensor.shape[0])

    # --- 提取并处理所有需要可视化的结果 ---
    V_pred_np = V_pred_tensor.cpu().numpy().flatten()

    rR_mask_p = torch.isclose(Xp[:, 1], torch.max(Xp[:, 1]))
    T_pred_np = best_model.T[rR_mask_p].cpu().numpy().flatten()

    Cp_pred_full_np = best_model.Cp.cpu().numpy()
    Cn_pred_full_np = best_model.Cn.cpu().numpy()

    rR_mask_n = torch.isclose(Xn[:, 1], torch.max(Xn[:, 1]))
    Cp_surf_pred_np = Cp_pred_full_np[rR_mask_p.cpu().numpy()]
    Cn_surf_pred_np = Cn_pred_full_np[rR_mask_n.cpu().numpy()]

    # --- 加载真值用于对比 ---
    true_data_pkg = pickle.load(open(datamodule.dataset.file_paths[0], 'rb'))
    true_results = true_data_pkg['results']
    time_array = true_results[SPM.time_col]
    V_true_np = Y_V_true_tensor.cpu().numpy().flatten()
    T_true_np = Y_T_true_tensor.cpu().numpy().flatten()
    Cp_surf_true_np = true_results[SPM.cp_surf_col]
    Cn_surf_true_np = true_results[SPM.cn_surf_col]

    # --- 开始绘图 ---
    print("\n--- 5. 正在生成结果对比图 ---")

    # 图1: 电压曲线
    plt.figure(figsize=(10, 6));
    plt.plot(time_array, V_true_np, 'b-', label="True Voltage");
    plt.plot(time_array, V_pred_np, 'r--', label="Predicted Voltage");
    plt.xlabel("Time [s]");
    plt.ylabel("Voltage [V]");
    plt.title("Terminal Voltage: True vs. Predicted");
    plt.legend();
    plt.grid(True);
    plt.savefig(f"{plots_dir}/voltage_comparison.png");
    plt.show();

    # 图2: 温度曲线
    plt.figure(figsize=(10, 6));
    plt.plot(time_array, T_true_np - 273.15, 'b-', label="True Temperature");
    plt.plot(time_array, T_pred_np - 273.15, 'r--', label="Predicted Temperature");
    plt.xlabel("Time [s]");
    plt.ylabel("Cell Temperature [°C]");
    plt.title("Cell Temperature: True vs. Predicted");
    plt.legend();
    plt.grid(True);
    plt.savefig(f"{plots_dir}/temperature_comparison.png");
    plt.show();

    # 图3: 表面浓度曲线
    fig, ax = plt.subplots(1, 2, figsize=(16, 6));
    fig.suptitle("Surface Concentration: True vs. Predicted");
    ax[0].plot(time_array, Cp_surf_true_np, 'b-', label="True");
    ax[0].plot(time_array, Cp_surf_pred_np, 'r--', label="Predicted");
    ax[0].set_title("Positive Electrode");
    ax[0].set_xlabel("Time [s]");
    ax[0].set_ylabel("Concentration [mol/m³]");
    ax[0].legend();
    ax[0].grid(True)
    ax[1].plot(time_array, Cn_surf_true_np, 'b-', label="True");
    ax[1].plot(time_array, Cn_surf_pred_np, 'r--', label="Predicted");
    ax[1].set_title("Negative Electrode");
    ax[1].set_xlabel("Time [s]");
    ax[1].set_ylabel("Concentration [mol/m³]");
    ax[1].legend();
    ax[1].grid(True)
    plt.savefig(f"{plots_dir}/surface_concentration.png");
    plt.show();

    # 图4: 浓度分布热图
    num_t_points = len(time_array)
    num_rp_points = len(true_results[SPM.rp_col][0])
    num_rn_points = len(true_results[SPM.rn_col][0])

    Cp_pred_grid = Cp_pred_full_np.reshape(num_rp_points, num_t_points)
    Cn_pred_grid = Cn_pred_full_np.reshape(num_rn_points, num_t_points)
    Cp_true_grid = np.stack(true_results[SPM.cp_col], axis=-1)
    Cn_true_grid = np.stack(true_results[SPM.cn_col], axis=-1)

    Rp = config["model_params"]["Rp"];
    Rn = config["model_params"]["Rn"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12));
    fig.suptitle("Particle Concentration Profile Heatmaps")
    im = axes[0, 0].imshow(Cp_true_grid, aspect='auto', origin='lower', extent=[0, time_array[-1], 0, Rp]);
    fig.colorbar(im, ax=axes[0, 0], label="Conc. [mol/m³]");
    axes[0, 0].set_title("True Positive Conc.");
    axes[0, 0].set_xlabel("Time [s]");
    axes[0, 0].set_ylabel("Radius [m]")
    im = axes[0, 1].imshow(Cp_pred_grid, aspect='auto', origin='lower', extent=[0, time_array[-1], 0, Rp]);
    fig.colorbar(im, ax=axes[0, 1], label="Conc. [mol/m³]");
    axes[0, 1].set_title("Predicted Positive Conc.");
    axes[0, 1].set_xlabel("Time [s]");
    axes[0, 1].set_ylabel("Radius [m]")
    im = axes[1, 0].imshow(Cn_true_grid, aspect='auto', origin='lower', extent=[0, time_array[-1], 0, Rn]);
    fig.colorbar(im, ax=axes[1, 0], label="Conc. [mol/m³]");
    axes[1, 0].set_title("True Negative Conc.");
    axes[1, 0].set_xlabel("Time [s]");
    axes[1, 0].set_ylabel("Radius [m]")
    im = axes[1, 1].imshow(Cn_pred_grid, aspect='auto', origin='lower', extent=[0, time_array[-1], 0, Rn]);
    fig.colorbar(im, ax=axes[1, 1], label="Conc. [mol/m³]");
    axes[1, 1].set_title("Predicted Negative Conc.");
    axes[1, 1].set_xlabel("Time [s]");
    axes[1, 1].set_ylabel("Radius [m]")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
    plt.savefig(f"{plots_dir}/concentration_heatmaps.png");
    plt.show();

    print(f"\n绘图完成！结果图已保存至 '{plots_dir}' 文件夹。")


if __name__ == "__main__":
    main()