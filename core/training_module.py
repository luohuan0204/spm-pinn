from typing import Any

import lightning.pytorch as pl
from torch import nn, optim
import torch


# 用于训练 SPM_PINN 模型
class TrainingModule(pl.LightningModule):
    """
    一个经过修改，可以支持学习率调度器的、更健壮的训练模块。
    """

    def __init__(
            self,
            model: nn.Module,
            loss_function: nn.Module,
            optimizer: optim.Optimizer,
            lr_scheduler=None,  # 修改点 1: 新增 lr_scheduler 参数，默认为None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_function', 'optimizer', 'lr_scheduler'])
        self.model = model
        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler  # 保存调度器

    # 定义训练数据集的单次训练步骤
    def training_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        """Step for training datasets."""

        # 保持您原来的解包逻辑
        I, Xp, Xn, Y, (N_t, _, _) = batch
        I, Xp, Xn, Y = I[0], Xp[0], Xn[0], Y[0]

        Y_pred = self.model(I, Xp, Xn, N_t)
        training_loss = self.loss_fn(Y_pred, Y, Xp, Xn, self.model)

        # 修改点 2: 使用更标准的日志名称 "train_loss"
        self.log("train_loss", training_loss, on_step=False, on_epoch=True, prog_bar=True)
        # (推荐) 同时记录学习率，方便观察
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True)

        return training_loss

    # 定义验证数据集的单次验证步骤
    def validation_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        """Step for validation datasets."""

        I, Xp, Xn, Y, (N_t, _, _) = batch
        I, Xp, Xn, Y = I[0], Xp[0], Xn[0], Y[0]

        Y_pred = self.model(I, Xp, Xn, N_t)
        # 将变量名改为 validation_loss 更贴切
        validation_loss = self.loss_fn(Y_pred, Y, Xp, Xn, self.model)

        # 修改点 3: 使用 "val_loss" 作为日志名称，这是调度器和模型保存监控的关键
        self.log("val_loss", validation_loss, on_step=False, on_epoch=True, prog_bar=True)
        return validation_loss

    # 定义测试数据集的单次测试步骤
    def test_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        """Step for test datasets."""

        I, Xp, Xn, Y, (N_t, _, _) = batch
        I, Xp, Xn, Y = I[0], Xp[0], Xn[0], Y[0]

        Y_pred = self.model(I, Xp, Xn, N_t)
        test_loss = self.loss_fn(Y_pred, Y, Xp, Xn, self.model)

        self.log("test_loss", test_loss)
        return test_loss

    def configure_optimizers(self) -> dict:  # 修改点 4: 重写此方法以支持调度器
        """Configure optimizer and learning rate scheduler."""

        if self.lr_scheduler is None:
            return self.optimizer
        else:
            # 返回PyTorch Lightning要求的标准字典格式
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.lr_scheduler,
                    "monitor": "train_loss",  # <-- 修改点
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

    # 修改点 5: 保持这两个方法不变，它们对于PINN在验证/测试时计算损失至关重要！
    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def on_test_model_eval(self, *args, **kwargs):
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)