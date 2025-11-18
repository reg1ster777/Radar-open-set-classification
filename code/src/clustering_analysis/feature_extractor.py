#!/usr/bin/env python3
"""
特征提取器
从预训练的backbone中提取特征用于聚类分析
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Tuple, List
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gcd_implementation.datasets import GCDDataset
from src.gcd_implementation.models import get_backbone


class FeatureExtractor:
    """
    特征提取器
    从预训练模型中提取特征向量
    """

    def __init__(self, backbone: nn.Module, device: torch.device = None):
        """
        初始化特征提取器

        Args:
            backbone: 预训练的骨干网络
            device: 计算设备
        """
        self.backbone = backbone.to(device)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.eval()  # 设置为评估模式

    def extract_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从数据集中提取特征

        Args:
            dataloader: 数据加载器

        Returns:
            features: 特征矩阵 [N, feature_dim]
            labels: 标签数组 [N]
            original_labels: 原始标签数组 [N]
        """
        all_features = []
        all_labels = []
        all_original_labels = []

        print("正在提取特征...")

        with torch.no_grad():
            for batch_idx, (signals, labels, original_labels) in enumerate(tqdm(dataloader, desc="提取特征")):
                signals = signals.to(self.device)

                # 提取特征
                features = self.backbone(signals)

                # 转换为numpy并移动到CPU
                features = features.cpu().numpy()
                labels = labels.cpu().numpy()
                original_labels = original_labels.cpu().numpy()

                all_features.append(features)
                all_labels.append(labels)
                all_original_labels.append(original_labels)

        # 合并所有批次
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        original_labels = np.concatenate(all_original_labels, axis=0)

        print(f"特征提取完成: {features.shape[0]} 个样本, {features.shape[1]} 维特征")

        return features, labels, original_labels

    @staticmethod
    def load_pretrained_backbone(model_path: str, arch_type: str = "ca1d",
                                input_channels: int = 1, device: torch.device = None) -> nn.Module:
        """
        加载预训练的骨干网络

        Args:
            model_path: 模型文件路径
            arch_type: 架构类型
            input_channels: 输入通道数
            device: 计算设备

        Returns:
            预训练的骨干网络
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建骨干网络
        backbone, actual_feature_dim = get_backbone(arch_type, input_channels=input_channels)
        print(f"骨干网络特征维度: {actual_feature_dim}")

        # 加载预训练权重
        if model_path and Path(model_path).exists():
            print(f"加载预训练模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)

            # 尝试不同的键名来加载backbone权重
            backbone_weights = None
            possible_keys = ['backbone', 'encoder', 'model.backbone', 'state_dict', 'G_state_dict']

            for key in possible_keys:
                if key in checkpoint:
                    backbone_weights = checkpoint[key]
                    print(f"找到权重键: {key}")
                    break

            if backbone_weights is None:
                # 如果找不到，尝试直接使用checkpoint
                backbone_weights = checkpoint
                print("使用整个checkpoint作为权重")

            # 加载权重，处理键名不匹配的情况
            try:
                backbone.load_state_dict(backbone_weights)
                print("✓ 预训练权重加载成功")
            except Exception as e:
                print(f"⚠️  直接加载失败，尝试逐层匹配: {e}")
                # 逐层匹配，跳过不匹配的层
                model_dict = backbone.state_dict()
                pretrained_dict = {k: v for k, v in backbone_weights.items() if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
                backbone.load_state_dict(model_dict)
                print(f"✓ 部分权重加载成功 ({len(pretrained_dict)}/{len(backbone_weights)})")
        else:
            print(f"⚠️  模型文件不存在: {model_path}")

        return backbone.to(device)


def create_dataloader_for_extraction(data_root: str, known_class_count: int = 7,
                                   batch_size: int = 256) -> Tuple[DataLoader, GCDDataset]:
    """
    创建用于特征提取的数据加载器

    Args:
        data_root: 数据根目录
        known_class_count: 已知类别数量
        batch_size: 批次大小

    Returns:
        dataloader: 数据加载器
        dataset: 数据集
    """
    # 创建数据集，包含所有类别
    dataset = GCDDataset(
        root=data_root,
        known_class_labels=list(range(1, known_class_count + 1)),
        unlabeled_class_labels=list(range(1, 11))  # 假设有10个类别
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 保持原始顺序以便后续分析
        num_workers=0
    )

    print(f"数据集加载完成: {len(dataset)} 个样本")

    return dataloader, dataset


if __name__ == "__main__":
    # 测试特征提取器
    data_root = "data/LFM_dataset/data_noise_50"
    model_path = "model/ca1d_gcd/pretrained_model.pth"  # 需要根据实际情况调整

    # 创建数据加载器
    dataloader, dataset = create_dataloader_for_extraction(data_root)

    # 加载预训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = FeatureExtractor.load_pretrained_backbone(model_path, device=device)

    # 创建特征提取器并提取特征
    extractor = FeatureExtractor(backbone, device)
    features, labels, original_labels = extractor.extract_features(dataloader)

    print(f"提取的特征形状: {features.shape}")
    print(f"标签范围: {labels.min()} - {labels.max()}")
    print(f"原始标签范围: {original_labels.min()} - {original_labels.max()}")