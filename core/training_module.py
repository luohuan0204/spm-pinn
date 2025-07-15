# core/training_module.py

from typing import Any, Dict
import lightning.pytorch as pl
from torch import nn, optim
import torch


class TrainingModule(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_function: nn.Module, optimizer: optim.Optimizer, lr_scheduler=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_function', 'optimizer', 'lr_scheduler'])
        self.model = model
        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def _common_step(self, batch: list, batch_idx: int, stage: str) -> Dict[str, Any]:

        # --- 【核心修正】: 直接解包batch，它本身就是一个包含5个张量的列表 ---
        I, Xp, Xn, Y_V_true, Y_T_true = batch

        # 因为batch_size=1, 每个张量都有一个多余的批次维度, 用.squeeze(0)去除
        I = I.squeeze(0)
        Xp = Xp.squeeze(0)
        Xn = Xn.squeeze(0)
        Y_V_true = Y_V_true.squeeze(0)
        Y_T_true = Y_T_true.squeeze(0)

        # 前向传播，得到预测电压
        V_pred = self.model(I, Xp, Xn, Y_V_true.shape[0])

        # 将Y_T_true传递给损失函数
        total_loss, loss_dict = self.loss_fn(V_pred, Y_V_true, Y_T_true, Xp, Xn, self.model)

        log_dict = {f"{stage}/{key}": value for key, value in loss_dict.items()}
        log_dict[f"{stage}/loss_total"] = total_loss
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=(stage == 'train'))

        return total_loss

    def training_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'], prog_bar=True)
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        # 即使在仅训练模式下，保留此方法也不会出错，Lightning会跳过它
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch: list, batch_idx: int) -> dict[str, Any]:
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self) -> dict:
        if self.lr_scheduler is None:
            return self.optimizer
        else:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.lr_scheduler,
                    "monitor": "train/loss_total",  # 在"仅训练"模式下监控训练损失
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def on_test_model_eval(self, *args, **kwargs):
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)