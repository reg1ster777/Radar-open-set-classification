import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Tuple, List
from collections import defaultdict

from src.gcd_implementation.datasets import GCDDataset


class GroundTruthSplitter:
    """
    基于真实标签的数据集分割器
    直接使用标签信息确定已知/未知类别，无需模型预测
    """

    def __init__(self, known_class_count: int):
        """
        初始化分割器

        Args:
            known_class_count: 已知类别数量
        """
        self.known_class_count = known_class_count

    def analyze_dataset(self, dataset: GCDDataset) -> dict:
        """
        分析数据集的标签分布

        Args:
            dataset: GCDDataset数据集

        Returns:
            analysis: 分析结果
        """
        print("分析数据集标签分布...")

        original_label_counts = defaultdict(int)
        for i in range(len(dataset)):
            _, _, original_label = dataset[i]
            original_label_counts[original_label] += 1

        print("数据集标签分布:")
        total_samples = sum(original_label_counts.values())

        known_count = 0
        unknown_count = 0

        for label in sorted(original_label_counts.keys()):
            count = original_label_counts[label]
            percentage = count / total_samples * 100

            if label <= self.known_class_count:
                print(f"  原始标签 {label} (已知类别): {count} 样本 ({percentage:.1f}%)")
                known_count += count
            else:
                print(f"  原始标签 {label} (未知类别): {count} 样本 ({percentage:.1f}%)")
                unknown_count += count

        print(f"\n总计:")
        print(f"  已知类别样本: {known_count} ({known_count/total_samples*100:.1f}%)")
        print(f"  未知类别样本: {unknown_count} ({unknown_count/total_samples*100:.1f}%)")
        print(f"  总样本数: {total_samples}")

        return {
            'original_label_counts': dict(original_label_counts),
            'total_samples': total_samples,
            'known_count': known_count,
            'unknown_count': unknown_count
        }

    def split_dataset_by_labels(self, dataset: GCDDataset,
                               labeled_ratio: float = 0.8) -> Tuple[Subset, Subset, List[int], List[int]]:
        """
        根据真实标签分割数据集

        Args:
            dataset: GCDDataset数据集
            labeled_ratio: 已知类别中有标签数据的比例

        Returns:
            labeled_dataset: 有标签数据集
            unlabeled_dataset: 无标签数据集
            known_indices: 已知类别样本索引
            unknown_indices: 未知类别样本索引
        """
        print("根据真实标签分割数据集...")

        known_indices = []
        unknown_indices = []
        labeled_indices = []

        # 遍历整个数据集，根据原始标签分类
        for i in range(len(dataset)):
            _, _, original_label = dataset[i]

            if original_label <= self.known_class_count:
                # 已知类别
                known_indices.append(i)
            else:
                # 未知类别
                unknown_indices.append(i)

        # 对已知类别进行GCD分割：80%有标签，20%无标签
        import random
        random.shuffle(known_indices)
        split_point = int(len(known_indices) * labeled_ratio)

        labeled_indices = known_indices[:split_point]
        unlabeled_known_indices = known_indices[split_point:]

        # 无标签数据 = 剩余的已知类别 + 所有未知类别
        unlabeled_indices = unlabeled_known_indices + unknown_indices

        # 创建数据集
        labeled_dataset = Subset(dataset, labeled_indices)
        unlabeled_dataset = Subset(dataset, unlabeled_indices)

        print(f"分割结果:")
        print(f"  有标签数据: {len(labeled_indices)} 样本")
        print(f"    - 已知类别: {len(labeled_indices)} 样本")
        print(f"  无标签数据: {len(unlabeled_indices)} 样本")
        print(f"    - 已知类别: {len(unlabeled_known_indices)} 样本")
        print(f"    - 未知类别: {len(unknown_indices)} 样本")
        print(f"  总计: {len(labeled_indices) + len(unlabeled_indices)} 样本")

        # 验证无重叠
        labeled_set = set(labeled_indices)
        unlabeled_set = set(unlabeled_indices)
        overlap = labeled_set.intersection(unlabeled_set)

        if overlap:
            print(f"⚠ 警告：发现 {len(overlap)} 个重叠索引")
        else:
            print("✓ 数据分割无重叠")

        return labeled_dataset, unlabeled_dataset, unknown_indices, known_indices

    def create_unknown_dataset(self, dataset: GCDDataset, unknown_indices: List[int]) -> Subset:
        """
        创建未知类别数据集

        Args:
            dataset: 原始数据集
            unknown_indices: 未知类别样本索引

        Returns:
            unknown_dataset: 未知类别数据集
        """
        print(f"创建未知类别数据集: {len(unknown_indices)} 样本")

        # 验证未知类别的真实性
        unknown_label_counts = defaultdict(int)
        for idx in unknown_indices:
            _, _, original_label = dataset[idx]
            unknown_label_counts[original_label] += 1

        print("未知类别数据集标签分布:")
        for label in sorted(unknown_label_counts.keys()):
            count = unknown_label_counts[label]
            print(f"  原始标签 {label}: {count} 样本")

        unknown_dataset = Subset(dataset, unknown_indices)
        return unknown_dataset

    def save_split_results(self, labeled_indices: List[int], unknown_indices: List[int],
                          known_indices: List[int], save_dir: Path):
        """
        保存分割结果

        Args:
            labeled_indices: 有标签数据索引
            unknown_indices: 未知类别数据索引
            known_indices: 已知类别数据索引
            save_dir: 保存目录
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'labeled_indices': labeled_indices,
            'unknown_indices': unknown_indices,
            'known_indices': known_indices,
            'known_class_count': self.known_class_count,
            'stats': {
                'labeled_count': len(labeled_indices),
                'unknown_count': len(unknown_indices),
                'known_count': len(known_indices)
            }
        }

        torch.save(results, save_dir / "ground_truth_split.pth")
        print(f"分割结果已保存到: {save_dir / 'ground_truth_split.pth'}")

    def load_split_results(self, load_dir: Path):
        """
        加载分割结果

        Args:
            load_dir: 加载目录

        Returns:
            results: 分割结果
        """
        results_path = load_dir / "ground_truth_split.pth"
        if results_path.exists():
            results = torch.load(results_path)
            print(f"从 {results_path} 加载分割结果")
            return results
        else:
            print(f"未找到 {results_path}，需要重新分割")
            return None