import sys
import os
from matplotlib import pyplot as plt
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

sys.path.append("../")
import lightning.pytorch as pl
from core.data_module import DataModule
from core.dataset import SimulationDataset
from core.loss import PINNLoss
from core.ocp_torch import get_graphite_ocp, get_nmc_ocp
from core.pinn_model import SPM_PINN
from core.physics_model import SPM
from core.training_module import TrainingModule
from torch import optim, vmap
import pickle
import torch
# 导入DataLoader
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # =============================================================================
    # 1. 配置文件
    # =============================================================================
    config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "train_params": {
            "max_epochs": 6000,  # 请务必设置一个足够大的值
            "learning_rate": 1e-4,
        },
        "data_params": {
            "data_directory": "../data",
        },
        "loss_weights": {
        "data": 5.0,                 # <-- 开启数据损失
        "ic": 1.0,                   # <-- 保留初始条件损失
        "bc_center": 0.00000001,
        "bc_surf": 0.000000005,             # <-- 开启表面边界损失
        "pde": 0.1,
        },
        "model_params": {
            "Up": vmap(get_nmc_ocp),
            "Cp_0": 17038.0, "Cp_max": 63104, "Rp": 5.22e-6, "ep_s": 0.335,
            "Lp": 75.6e-6, "kp": 5e-10, "Dp": 1e-14,
            "Un": vmap(get_graphite_ocp),
            "Cn_0": 29866.0, "Cn_max": 33133, "Rn": 5.22e-6, "en_s": 0.75,
            "Ln": 75.6e-6, "kn": 5e-10, "Dn": 3e-14,
            "Ce": 1000, "R_cell": 3.24e-4,
            "nn_hidden_size": 128,
            "nn_num_hidden_layers": 8,
        }
    }

    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    # =============================================================================
    # 2. 初始化模块
    # =============================================================================

    # 数据加载 (只创建训练加载器)
    dataset = SimulationDataset(data_directory=config["data_params"]["data_directory"])
    train_loader = DataLoader(dataset, batch_size=1)

    # 模型初始化
    model = SPM_PINN(**config["model_params"]).to(config["device"])

    # 损失函数和优化器
    loss_fn = PINNLoss(weights=config["loss_weights"], T_norm=1000, R_norm=5.22e-6)
    optimizer = optim.Adam(params=model.parameters(), lr=config["train_params"]["learning_rate"])

    # 学习率调度器现在监控 train_loss
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=500, verbose=True)

    training_module = TrainingModule(
        model=model,
        loss_function=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # =============================================================================
    # 3. 设置训练器并开始训练
    # =============================================================================
    logger = TensorBoardLogger("results/logs", name="spm_pinn_train_only")

    # ModelCheckpoint 现在也监控 train_loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="results/checkpoints",
        filename="best_train_model-{epoch:02d}-{train_loss:.2e}",
        save_top_k=1,
        verbose=True,
        monitor="train_loss",  # <-- 修改点
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=config["train_params"]["max_epochs"],
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # --- 新增：指定从第一阶段训练好的模型继续训练 ---
    # 请确保文件名与您第一阶段训练保存的最佳模型文件名完全一致
    # 根据您上次的日志，应该是 'best_train_model-epoch=4922-train_loss=2.61e-10.ckpt'
    first_stage_ckpt_path = "results/pretrained_models/best_train_model-epoch=4922-train_loss=2.61e-10.ckpt"

    # 检查文件是否存在，避免报错
    if not os.path.exists(first_stage_ckpt_path):
        raise FileNotFoundError(
            f"错误：找不到第一阶段的检查点文件: {first_stage_ckpt_path}\n"
            f"请先运行第一阶段的训练，或检查文件名是否正确。"
        )
    else:
        print(f"成功找到第一阶段模型，将从 {first_stage_ckpt_path} 继续训练...")

    # --- 修改训练调用，加入 ckpt_path 参数 ---
    trainer.fit(
        model=training_module,
        train_dataloaders=train_loader,
        ckpt_path=first_stage_ckpt_path  # <-- 核心改动在这里
    )

    # =============================================================================
    # 4. 加载最佳模型并进行预测和绘图
    # =============================================================================
    print("训练完成，加载最佳模型进行预测...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        # 从检查点加载训练模块，然后获取模型
        training_module_loaded = TrainingModule.load_from_checkpoint(
            best_model_path,
            model=model,
            loss_function=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )
        best_model = training_module_loaded.model
        print(f"从 {best_model_path} 加载了最佳模型。")
    else:
        # 如果没有保存任何模型，则使用训练结束时的模型
        print("没有找到检查点文件，将使用训练结束时的最终模型。")
        best_model = model

    best_model.eval()
    best_model.to(config["device"])

    # 准备预测所需的全部数据点
    I, Xp, Xn, Y, (N_t, N_rp, N_rn) = dataset[0]
    I, Xp, Xn, Y = I.to(config["device"]), Xp.to(config["device"]), Xn.to(config["device"]), Y.to(config["device"])

    # 运行一次完整的预测
    with torch.no_grad():
        voltage_pred_tensor = best_model(I, Xp, Xn, N_t)

        # 提取所有需要绘图的数据
        # detach()用于切断梯度，cpu()用于移到cpu，numpy()用于转成numpy数组
        voltage_pred = voltage_pred_tensor.cpu().numpy()[:, 0]

        # 提取初始浓度 (t=0, 归一化坐标为-1)
        Cp0_pred = best_model.Cp[Xp[:, 0] == -1].cpu().numpy()[:, 0]
        Cn0_pred = best_model.Cn[Xn[:, 0] == -1].cpu().numpy()[:, 0]

        # 提取最终浓度 (t=T, 归一化坐标为1)
        Cplast_pred = best_model.Cp[Xp[:, 0] == 1].cpu().numpy()[:, 0]
        Cnlast_pred = best_model.Cn[Xn[:, 0] == 1].cpu().numpy()[:, 0]

        # 提取表面浓度随时间的变化
        Cp_surf_pred = best_model.Cp_clamped[-N_t:].cpu().numpy()[:, 0]  # 使用裁剪后的浓度更准确
        Cn_surf_pred = best_model.Cn_clamped[-N_t:].cpu().numpy()[:, 0]

    # 加载真实数据用于对比
    with open("../data/spm0_Dp=1e-14_Dn=3e-14", "rb") as binary_file:
        true_data = pickle.load(binary_file)

    # =============================================================================
    # 5. 绘制所有对比图
    # =============================================================================
    print("正在生成结果对比图...")

    # 图1: 端电压 vs. 时间
    plt.figure(figsize=(10, 6))
    plt.plot(true_data.get(SPM.time_col), true_data.get(SPM.voltage_col), label="True")
    plt.plot(true_data.get(SPM.time_col), voltage_pred, label="Predicted")
    plt.xlabel(SPM.time_col)
    plt.ylabel(SPM.voltage_col)
    plt.title("Terminal Voltage")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/terminal_voltage_final.png")
    plt.show()
    plt.close()

    # 图2: 正极表面浓度 vs. 时间
    plt.figure(figsize=(10, 6))
    plt.plot(true_data.get(SPM.time_col), true_data.get(SPM.cp_surf_col), label="True")
    plt.plot(true_data.get(SPM.time_col), Cp_surf_pred, label="Predicted")
    plt.xlabel(SPM.time_col)
    plt.ylabel(SPM.cp_surf_col)
    plt.title("Positive electrode surface concentration")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/positive_surface_concentration.png")
    plt.show()
    plt.close()

    # 图3: 负极表面浓度 vs. 时间
    plt.figure(figsize=(10, 6))
    plt.plot(true_data.get(SPM.time_col), true_data.get(SPM.cn_surf_col), label="True")
    plt.plot(true_data.get(SPM.time_col), Cn_surf_pred, label="Predicted")
    plt.xlabel(SPM.time_col)
    plt.ylabel(SPM.cn_surf_col)
    plt.title("Negative electrode surface concentration")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/negative_surface_concentration.png")
    plt.show()
    plt.close()

    # 图4: 初始浓度 vs. 半径
    plt.figure(figsize=(10, 6))
    plt.plot(true_data.get(SPM.rp_col)[0], Cp0_pred, label="Predicted positive")
    plt.plot(true_data.get(SPM.rn_col)[0], Cn0_pred, label="Predicted negative")
    plt.plot(true_data.get(SPM.rp_col)[0], true_data.get(SPM.cp_col)[0], label="True positive")
    plt.plot(true_data.get(SPM.rn_col)[0], true_data.get(SPM.cn_col)[0], label="True negative")
    plt.xlabel(f"{SPM.rp_col} | {SPM.rn_col}")
    plt.ylabel("Concentration [mol/m3]")
    plt.title("Initial concentration")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/initial_concentration.png")
    plt.show()
    plt.close()

    # 图5: 最终浓度 vs. 半径
    plt.figure(figsize=(10, 6))
    plt.plot(true_data.get(SPM.rp_col)[-1], Cplast_pred, label="Predicted positive")
    plt.plot(true_data.get(SPM.rn_col)[-1], Cnlast_pred, label="Predicted negative")
    plt.plot(true_data.get(SPM.rp_col)[-1], true_data.get(SPM.cp_col)[-1], label="True positive")
    plt.plot(true_data.get(SPM.rn_col)[-1], true_data.get(SPM.cn_col)[-1], label="True negative")
    plt.xlabel(f"{SPM.rp_col} | {SPM.rn_col}")
    plt.ylabel("Concentration [mol/m3]")
    plt.title("Final concentration")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/final_concentration.png")
    plt.show()
    plt.close()

    print("所有图像已生成并保存。")
