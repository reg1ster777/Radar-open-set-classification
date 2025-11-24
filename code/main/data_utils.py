"""数据准备与 DataLoader 构建模块。

负责从原始数据集中提取开集训练/测试样本，并提供基础的张量预处理工具。
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.solver.dataset import load_radar_known_fixed_test, load_radar_unknown
import config


def create_open_set_dataset() -> Dict[str, torch.Tensor]:
    """加载开集所需的已知/未知数据，并按照既定规则划分。"""
    print("创建开集数据集...")
    train_loader_known, test_loader_known = load_radar_known_fixed_test(
        root=str(config.DATA_ROOT),
        known_class_count=config.KNOWN_CLASS_COUNT,
        test_per_class=1000,
        batch_size=256,
        shuffle_train=False,
        shuffle_test=False,
        as_loader=True,
    )

    unknown_loader = load_radar_unknown(
        root=str(config.DATA_ROOT),
        known_class_count=config.KNOWN_CLASS_COUNT,
        batch_size=256,
        as_loader=True,
    )

    train_known_data = torch.cat([data for data, _ in train_loader_known], dim=0)
    train_known_labels = torch.cat([labels for _, labels in train_loader_known], dim=0)

    test_known_data = torch.cat([data for data, _ in test_loader_known], dim=0)
    test_known_labels = torch.cat([labels for _, labels in test_loader_known], dim=0)

    unknown_data = torch.cat([data for data, _ in unknown_loader], dim=0)

    print(f"  已知类训练样本: {len(train_known_data)}")
    print(f"  已知类测试样本: {len(test_known_data)}")
    print(f"  未知类总样本: {len(unknown_data)}")

    samples_per_unknown_class = len(unknown_data) // config.UNKNOWN_CLASS_COUNT
    unknown_train_data, unknown_train_labels = [], []
    unknown_test_data, unknown_test_labels = [], []

    for class_idx in range(config.UNKNOWN_CLASS_COUNT):
        start_idx = class_idx * samples_per_unknown_class
        end_idx = start_idx + samples_per_unknown_class
        class_data = unknown_data[start_idx:end_idx]
        indices = torch.randperm(len(class_data))
        train_idx = indices[:4000]
        test_idx = indices[4000:5000]

        unknown_train_data.append(class_data[train_idx])
        unknown_test_data.append(class_data[test_idx])

        class_label = config.KNOWN_CLASS_COUNT + class_idx
        unknown_train_labels.append(torch.full((len(train_idx),), class_label, dtype=torch.long))
        unknown_test_labels.append(torch.full((len(test_idx),), class_label, dtype=torch.long))

    dataset = {
        'train_known_data': train_known_data,
        'train_known_labels': train_known_labels,
        'test_known_data': test_known_data,
        'test_known_labels': test_known_labels,
        'train_unknown_data': torch.cat(unknown_train_data, dim=0),
        'train_unknown_labels': torch.cat(unknown_train_labels, dim=0),
        'test_unknown_data': torch.cat(unknown_test_data, dim=0),
        'test_unknown_labels': torch.cat(unknown_test_labels, dim=0),
    }
    print(f"  未知类训练样本: {len(dataset['train_unknown_data'])} (每类4000)")
    print(f"  未知类测试样本: {len(dataset['test_unknown_data'])} (每类1000)")
    return dataset


def ensure_channel_dim(signals: torch.Tensor) -> torch.Tensor:
    """确保输入张量包含显式的通道维。"""
    if signals.dim() == 2:
        return signals.unsqueeze(1)
    return signals


def build_supervised_loader(signals: torch.Tensor, labels: torch.Tensor, batch_size: int = 256, shuffle: bool = True) -> DataLoader:
    """构建监督训练/评估所需的 DataLoader。"""
    dataset = TensorDataset(signals, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
