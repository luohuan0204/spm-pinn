# core/data_module.py

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class DataModule(L.LightningDataModule):
    """
    一个健壮的、可复现的数据模块，可将数据集划分为训练、验证和测试集。
    """

    def __init__(
            self,
            dataset: Dataset,
            train_split: float = 0.7,
            val_split: float = 0.2,
            batch_size: int = 1,
            seed: int = 42,
    ) -> None:
        super().__init__()

        # --- 【最终修正】: 放宽检查条件，允许 train_split + val_split 等于 1 ---
        # 检查划分比例是否在[0, 1]范围内，并且它们的和不大于1
        if not (0 <= train_split <= 1 and 0 <= val_split <= 1 and (train_split + val_split) <= 1):
            raise ValueError("train_split and val_split must be between 0 and 1, and their sum must not exceed 1.")

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1.0 - self.train_split - self.val_split

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, stage: str) -> None:
        # 使用更稳健的划分方法，确保总数正确
        num_dataset = len(self.dataset)

        # 即使 val_split 为 0，int(0.0 * num_dataset) 也会正确地得到 0
        num_validation_set = int(self.val_split * num_dataset)
        num_test_set = int(self.test_split * num_dataset)
        num_training_set = num_dataset - num_validation_set - num_test_set

        # 使用带种子的生成器来确保划分的可复现性
        generator = torch.Generator().manual_seed(self.seed)
        self.training_set, self.validation_set, self.test_set = random_split(
            self.dataset,
            [num_training_set, num_validation_set, num_test_set],
            generator=generator
        )
        print(
            f"Dataset split: Train={len(self.training_set)}, Val={len(self.validation_set)}, Test={len(self.test_set)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_set, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size)