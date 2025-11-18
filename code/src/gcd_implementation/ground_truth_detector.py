import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

from src.gcd_implementation.datasets import GCDDataset


class GroundTruthUnknownDetector:
    """
    基于真实标签的未知样本检测器
    用于调试和对比，直接使用真实标签来分离未知样本
    """

    def __init__(self, known_class_count: int):
        """
        初始化基于真实标签的检测器

        Args:
            known_class_count: 已知类别数量
        """
        self.known_class_count = known_class_count

    def detect_unknown_samples_ground_truth(self, unlabeled_dataloader: DataLoader) -> Tuple[List[int], List[int]]:
        """
        基于真实标签检测未知样本

        Args:
            unlabeled_dataloader: 无标签数据加载器

        Returns:
            unknown_indices: 未知样本的索引列表
            known_indices: 已知样本的索引列表
        """
        print("基于真实标签检测未知样本...")

        unknown_indices = []
        known_indices = []

        all_original_labels = []

        # 收集所有原始标签
        for batch_idx, (signals, _, original_labels) in enumerate(unlabeled_dataloader):
            for i, original_label in enumerate(original_labels):
                global_idx = batch_idx * unlabeled_dataloader.batch_size + i
                all_original_labels.append(original_label.item())

                # 根据真实标签分类
                if original_label.item() > self.known_class_count:
                    unknown_indices.append(global_idx)
                else:
                    known_indices.append(global_idx)

        # 统计分析
        total_samples = len(all_original_labels)
        unknown_count = len(unknown_indices)
        known_count = len(known_indices)

        print(f"基于真实标签的检测结果:")
        print(f"  总样本数: {total_samples}")
        print(f"  真实未知样本: {unknown_count} ({unknown_count/total_samples*100:.1f}%)")
        print(f"  真实已知样本: {known_count} ({known_count/total_samples*100:.1f}%)")

        # 详细标签分布
        label_counts = defaultdict(int)
        for label in all_original_labels:
            label_counts[label] += 1

        print(f"\n真实标签分布:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            if label <= self.known_class_count:
                print(f"  已知类别 {label}: {count} 样本")
            else:
                print(f"  未知类别 {label}: {count} 样本")

        return unknown_indices, known_indices

    def create_balanced_unknown_dataset(self, unlabeled_dataloader: DataLoader,
                                      target_unknown_ratio: float = 0.3) -> Tuple[List[int], List[int]]:
        """
        创建平衡的未知样本检测结果

        Args:
            unlabeled_dataloader: 无标签数据加载器
            target_unknown_ratio: 目标未知样本比例

        Returns:
            unknown_indices: 未知样本的索引列表
            known_indices: 已知样本的索引列表
        """
        unknown_indices, known_indices = self.detect_unknown_samples_ground_truth(unlabeled_dataloader)

        # 如果未知样本比例太低，进行下采样
        total_samples = len(unknown_indices) + len(known_indices)
        current_unknown_ratio = len(unknown_indices) / total_samples

        if current_unknown_ratio < target_unknown_ratio:
            print(f"当前未知样本比例 {current_unknown_ratio:.1%} 低于目标 {target_unknown_ratio:.1%}")
            print("对已知样本进行下采样...")

            # 计算需要的已知样本数量
            target_known_count = int(len(unknown_indices) * (1 - target_unknown_ratio) / target_unknown_ratio)

            if len(known_indices) > target_known_count:
                import random
                random.shuffle(known_indices)
                known_indices = known_indices[:target_known_count]

        final_total = len(unknown_indices) + len(known_indices)
        final_unknown_ratio = len(unknown_indices) / final_total

        print(f"平衡后结果:")
        print(f"  未知样本: {len(unknown_indices)} ({final_unknown_ratio:.1%})")
        print(f"  已知样本: {len(known_indices)} ({1-final_unknown_ratio:.1%})")

        return unknown_indices, known_indices