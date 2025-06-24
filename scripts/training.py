import sys
import os
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建结果目录
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/checkpoints", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

# 数据加载
dataset = SimulationDataset(data_directory="../data")
data_module = DataModule(dataset=dataset, train_split=1, val_split=0, test_split=0)

# 模型初始化
model = SPM_PINN(
    Up=vmap(get_nmc_ocp),
    Cp_0=35263,
    Cp_max=63104,
    Rp=5.22e-6,
    ep_s=0.335,
    Lp=75.6e-6,
    kp=5e-10,
    Dp=1e-14,
    Un=vmap(get_graphite_ocp),
    Cn_0=15528,
    Cn_max=33133,
    Rn=5.22e-6,
    en_s=0.75,
    Ln=75.6e-6,
    kn=5e-10,
    Dn=3e-14,
    Ce=1000,
    R_cell=3.24e-4,
    nn_hidden_size=64,
    nn_num_hidden_layers=12,
).to(device)

# 损失函数和优化器
loss_fn = PINNLoss()
optimizer_algorithm = optim.Adam
optimizer_kwargs = {"lr": 1e-4}
optimizer = optimizer_algorithm(params=model.parameters(), **optimizer_kwargs)

# 训练模块
training_module = TrainingModule(
    model=model,
    loss_function=loss_fn,
    optimizer=optimizer,
)

# 训练器
logger = TensorBoardLogger("results/logs", name="spm_pinn")
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_epochs=100,
    logger=True,
    enable_progress_bar=True,
    enable_model_summary=False,
    enable_checkpointing=False,
)

# 训练模型
trainer.fit(training_module, data_module)

# 模型预测
model.eval()
model.to(device)
I, Xp, Xn, Y, (N_t, N_rp, N_rn) = dataset[0]
I, Xp, Xn, Y = I.to(device), Xp.to(device), Xn.to(device), Y.to(device)
out = model(I, Xp, Xn, N_t)

Cp0 = model.Cp[Xp[:, 0] == -1].cpu().detach().numpy()[:, 0]
Cn0 = model.Cn[Xn[:, 0] == -1].cpu().detach().numpy()[:, 0]
Cplast = model.Cp[Xp[:, 0] == 1].cpu().detach().numpy()[:, 0]
Cnlast = model.Cn[Xn[:, 0] == 1].cpu().detach().numpy()[:, 0]
Cp_surf = model.Cp[-N_t:].cpu().detach().numpy()[:, 0]
Cn_surf = model.Cn[-N_t:].cpu().detach().numpy()[:, 0]

# 加载真实数据用于对比
with open("../data/spm0_Dp=1e-14_Dn=3e-14", "rb") as binary_file:
    true_data = pickle.load(binary_file)

# =========== Plots ================
# 使用 Matplotlib 替换原始的 plot 函数调用
# 1. 初始浓度
plt.figure(figsize=(10, 6))
plt.plot(true_data.get(SPM.rp_col)[0], Cp0, label="Predicted positive", linestyle='-')
plt.plot(true_data.get(SPM.rn_col)[0], Cn0, label="Predicted negative", linestyle='-')
plt.plot(true_data.get(SPM.rp_col)[0], true_data.get(SPM.cp_col)[0], label="True positive", linestyle='-')
plt.plot(true_data.get(SPM.rn_col)[0], true_data.get(SPM.cn_col)[0], label="True negative", linestyle='-')
plt.xlabel(f"{SPM.rp_col} | {SPM.rn_col}")
plt.ylabel(f"{SPM.cp_col} | {SPM.cn_col}")
plt.title("Initial concentration")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/initial_concentration.png")
plt.show()
plt.close()

# 2. 最终浓度
plt.figure(figsize=(10, 6))
plt.plot(true_data.get(SPM.rp_col)[-1], Cplast, label="Predicted positive", linestyle='-')
plt.plot(true_data.get(SPM.rn_col)[-1], Cnlast, label="Predicted negative", linestyle='-')
plt.plot(true_data.get(SPM.rp_col)[-1], true_data.get(SPM.cp_col)[-1], label="True positive", linestyle='-')
plt.plot(true_data.get(SPM.rn_col)[-1], true_data.get(SPM.cn_col)[-1], label="True negative", linestyle='-')
plt.xlabel(f"{SPM.rp_col} | {SPM.rn_col}")
plt.ylabel(f"{SPM.cp_col} | {SPM.cn_col}")
plt.title("Final concentration")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/final_concentration.png")
plt.show()
plt.close()

# 3. 正极表面浓度
plt.figure(figsize=(10, 6))
plt.plot(true_data.get(SPM.time_col), true_data.get(SPM.cp_surf_col), label="True", linestyle='-')
plt.plot(true_data.get(SPM.time_col), Cp_surf, label="Predicted", linestyle='-')
plt.xlabel(SPM.time_col)
plt.ylabel(SPM.cp_surf_col)
plt.title("Positive electrode surface concentration")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/positive_surface_concentration.png")
plt.show()
plt.close()

# 4. 负极表面浓度
plt.figure(figsize=(10, 6))
plt.plot(true_data.get(SPM.time_col), true_data.get(SPM.cn_surf_col), label="True", linestyle='-')
plt.plot(true_data.get(SPM.time_col), Cn_surf, label="Predicted", linestyle='-')
plt.xlabel(SPM.time_col)
plt.ylabel(SPM.cn_surf_col)
plt.title("Negative electrode surface concentration")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/negative_surface_concentration.png")
plt.show()
plt.close()

# 5. 端电压
plt.figure(figsize=(10, 6))
plt.plot(true_data.get(SPM.time_col), true_data.get(SPM.voltage_col), label="True", linestyle='-')
plt.plot(true_data.get(SPM.time_col), out.cpu().detach().numpy()[:, 0], label="Predicted", linestyle='-')
plt.xlabel(SPM.time_col)
plt.ylabel(SPM.voltage_col)
plt.title("Terminal Voltage")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/terminal_voltage.png")
plt.show()
plt.close()