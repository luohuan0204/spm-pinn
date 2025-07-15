# core/dataset.py

import os
import pickle
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from core.physics_model import SPM


class SimulationDataset(Dataset):
    """
    加载数据，动态归一化，并为模型提供电压和温度的真值。
    """

    def __init__(self, data_directory: str):
        self.file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.pkl')]
        self.data_cache = {}

    def __len__(self) -> int:
        return len(self.file_paths)

    def get_params(self, index: int) -> dict:
        file_path = self.file_paths[index]
        with open(file_path, "rb") as binary_file:
            data_package = pickle.load(binary_file)
        params = data_package['parameters']
        results = data_package['results']
        time_data = np.array(results[SPM.time_col])
        params['duration'] = time_data.max()
        return params

    def __getitem__(self, index: int) -> tuple:
        if index in self.data_cache:
            return self.data_cache[index]

        file_path = self.file_paths[index]
        with open(file_path, "rb") as binary_file:
            data_package = pickle.load(binary_file)

        params = data_package['parameters']
        results = data_package['results']

        normalized_results = self.normalize_data(results, params)

        time_norm = normalized_results[SPM.time_col]
        rp_norm = normalized_results[SPM.rp_col][0]
        rn_norm = normalized_results[SPM.rn_col][0]

        t_grid_p, rp_grid = np.meshgrid(time_norm, rp_norm)
        t_grid_n, rn_grid = np.meshgrid(time_norm, rn_norm)

        # --- 准备输入和标签 ---
        I = torch.tensor(results[SPM.current_col], dtype=torch.float32).view(-1, 1)
        Xp = torch.stack([torch.tensor(t_grid_p.flatten(), dtype=torch.float32),
                          torch.tensor(rp_grid.flatten(), dtype=torch.float32)], axis=-1)
        Xn = torch.stack([torch.tensor(t_grid_n.flatten(), dtype=torch.float32),
                          torch.tensor(rn_grid.flatten(), dtype=torch.float32)], axis=-1)

        # 标签Y_V是电压的真值
        Y_V_true = torch.tensor(results[SPM.voltage_col], dtype=torch.float32).view(-1, 1)

        # --- 【核心修改】: 新增温度标签 Y_T_true ---
        # 从结果中提取温度，并去掉第一个点以和时间对齐
        T_true_array = np.array(results[SPM.temp_col])[1:]
        Y_T_true = torch.tensor(T_true_array, dtype=torch.float32).view(-1, 1)

        Xp.requires_grad = True
        Xn.requires_grad = True

        # 返回一个包含5个元素的元组
        processed_data = (I, Xp, Xn, Y_V_true, Y_T_true)
        self.data_cache[index] = processed_data

        return processed_data

    def normalize_data(self, results: dict, params: dict) -> dict:
        # ... 此函数保持不变 ...
        normalized = {}
        time_data = np.array(results[SPM.time_col])
        t_min, t_max = time_data.min(), time_data.max()
        time_span = t_max - t_min if t_max > t_min else 1.0
        normalized[SPM.time_col] = (time_data - t_min) / time_span * 2 - 1
        rp_data = np.array(results[SPM.rp_col])
        rp_max = params['Rp']
        normalized[SPM.rp_col] = (rp_data / rp_max) * 2 - 1
        rn_data = np.array(results[SPM.rn_col])
        rn_max = params['Rn']
        normalized[SPM.rn_col] = (rn_data / rn_max) * 2 - 1
        return normalized