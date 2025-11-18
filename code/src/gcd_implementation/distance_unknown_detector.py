import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Tuple, List
from collections import defaultdict

from src.gcd_implementation.datasets import GCDDataset


class DistanceBasedUnknownDetector:
    """
    基于距离阈值的未知类别检测器
    参考test_openset_recognition.py中的类中心距离方法
    """

    def __init__(self, backbone, classification_head, device, known_class_count=7):
        """
        初始化基于距离的未知类别检测器

        Args:
            backbone: 特征提取backbone
            classification_head: 分类头
            device: 计算设备
            known_class_count: 已知类别数量
        """
        self.backbone = backbone
        self.classification_head = classification_head
        self.device = device
        self.known_class_count = known_class_count

        self.backbone.eval()
        self.classification_head.eval()

        self.class_centroids = None
        self.global_threshold = None
        self.class_thresholds = None

    def extract_features_and_labels(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        提取特征和标签

        Args:
            dataloader: 数据加载器

        Returns:
            features: 特征矩阵 [N, feature_dim]
            labels: 标签 [N] (1-based)
            predictions: 预测结果 [N] (1-based)
        """
        all_features = []
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for signals, labels, _ in dataloader:
                signals = signals.to(self.device)

                # 前向传播获取特征
                features = self.backbone(signals)

                # 获取预测
                logits = self.classification_head(features)
                predictions = torch.argmax(logits, dim=1) + 1  # 转换为1-based

                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        predictions = np.concatenate(all_predictions, axis=0)

        return features, labels, predictions

    def compute_cosine_distance(self, features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        计算余弦距离

        Args:
            features: 特征矩阵 [N, feature_dim]
            centroids: 中心矩阵 [K, feature_dim]

        Returns:
            distances: 距离矩阵 [N, K]
        """
        # 归一化
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

        # 计算余弦相似度
        similarities = np.dot(features_norm, centroids_norm.T)

        # 转换为距离
        distances = 1.0 - similarities
        return distances

    def fit_known_classes(self, labeled_dataloader: DataLoader):
        """
        在有标签数据上拟合已知类别的中心和阈值

        Args:
            labeled_dataloader: 有标签数据加载器
        """
        print("计算已知类别的中心和阈值...")

        # 提取有标签数据的特征和标签
        features, labels, predictions = self.extract_features_and_labels(labeled_dataloader)

        # 只使用已知类别的数据（标签1-7）
        known_mask = labels <= self.known_class_count
        known_features = features[known_mask]
        known_labels = labels[known_mask] - 1  # 转换为0-based索引

        print(f"使用 {len(known_features)} 个已知样本计算中心")

        # 计算每个已知类别的中心
        num_classes = self.known_class_count
        self.class_centroids = np.zeros((num_classes, known_features.shape[1]))

        for i in range(num_classes):
            class_mask = known_labels == i
            if np.sum(class_mask) > 0:
                class_features = known_features[class_mask]
                centroid = class_features.mean(axis=0)
                # 归一化中心
                self.class_centroids[i] = centroid / (np.linalg.norm(centroid) + 1e-8)
            else:
                # 如果没有样本，使用零向量
                self.class_centroids[i] = np.zeros(known_features.shape[1])

        # 计算每个样本到其类别中心的距离
        distances = self.compute_cosine_distance(known_features, self.class_centroids)

        # 计算每个样本到其所属类别的距离
        self_distances = distances[np.arange(len(known_labels)), known_labels]

        # 计算全局阈值（95%分位数）
        self.global_threshold = np.quantile(self_distances, 0.95)

        # 计算每个类别的阈值
        self.class_thresholds = np.zeros(num_classes, dtype=float)
        for i in range(num_classes):
            class_mask = known_labels == i
            if np.sum(class_mask) > 0:
                # 修复索引错误
                class_distances = distances[class_mask, i]
                self.class_thresholds[i] = np.quantile(class_distances, 0.95)
            else:
                self.class_thresholds[i] = self.global_threshold

        print(f"全局距离阈值: {self.global_threshold:.6f}")
        print(f"各类别阈值: {self.class_thresholds}")

    def detect_unknown_samples(self, unlabeled_dataloader: DataLoader,
                              use_global_threshold: bool = True,
                              use_class_thresholds: bool = True,
                              tail_size: float = 0.05) -> Tuple[List[int], List[int]]:
        """
        使用距离阈值检测未知样本

        Args:
            unlabeled_dataloader: 无标签数据加载器
            use_global_threshold: 是否使用全局阈值
            use_class_thresholds: 是否使用类别阈值
            tail_size: 用于调整阈值的尾部比例

        Returns:
            unknown_indices: 未知样本的索引列表
            known_indices: 已知样本的索引列表
        """
        print("使用距离阈值检测未知样本...")

        if self.class_centroids is None:
            raise ValueError("请先调用 fit_known_classes() 方法计算已知类别中心")

        # 提取无标签数据的特征和预测
        features, labels, predictions = self.extract_features_and_labels(unlabeled_dataloader)

        # 计算到所有已知类别中心的距离
        distances = self.compute_cosine_distance(features, self.class_centroids)

        # 找到最近的已知类别和距离
        min_distances = np.min(distances, axis=1)
        nearest_classes = np.argmin(distances, axis=1) + 1  # 转换为1-based

        # 使用预测结果进行初步筛选
        predicted_known_mask = predictions <= self.known_class_count

        # 方法1: 使用全局阈值
        if use_global_threshold:
            global_unknown_mask = min_distances > self.global_threshold
        else:
            global_unknown_mask = np.zeros(len(features), dtype=bool)

        # 方法2: 使用类别特定阈值
        if use_class_thresholds:
            class_unknown_mask = np.zeros(len(features), dtype=bool)
            for i in range(len(features)):
                predicted_class = predictions[i] - 1  # 转换为0-based
                if 0 <= predicted_class < self.known_class_count:
                    class_threshold = self.class_thresholds[predicted_class]
                    class_unknown_mask[i] = distances[i, predicted_class] > class_threshold
                else:
                    # 如果预测为未知类别，直接标记为未知
                    class_unknown_mask[i] = True
        else:
            class_unknown_mask = np.zeros(len(features), dtype=bool)

        # 方法3: 结合预测置信度（使用预测概率的熵）
        # 这里使用预测分布的平滑度作为辅助判断
        with torch.no_grad():
            signals_batch = []
            for batch_signals, _, _ in unlabeled_dataloader:
                signals_batch.append(batch_signals.to(self.device))

            all_logits = []
            for batch_signals in signals_batch:
                features_batch = self.backbone(batch_signals)
                logits_batch = self.classification_head(features_batch)
                all_logits.append(logits_batch.cpu())

            all_logits = torch.cat(all_logits, dim=0)
            probabilities = F.softmax(all_logits, dim=1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)

            # 高熵表示预测不确定
            entropy_threshold = np.quantile(entropy.numpy(), 0.8)
            entropy_unknown_mask = entropy.numpy() > entropy_threshold

        # 综合判断
        final_unknown_mask = (
            (predicted_known_mask & (global_unknown_mask | class_unknown_mask)) |
            (~predicted_known_mask & entropy_unknown_mask)
        )

        # 收集索引
        unknown_indices = []
        known_indices = []

        for i in range(len(features)):
            if final_unknown_mask[i]:
                unknown_indices.append(i)
            else:
                known_indices.append(i)

        # 输出统计信息
        total_samples = len(features)
        unknown_count = len(unknown_indices)
        known_count = len(known_indices)

        print(f"\n距离阈值检测结果:")
        print(f"  总样本数: {total_samples}")
        print(f"  检测为未知: {unknown_count} ({unknown_count/total_samples*100:.1f}%)")
        print(f"  检测为已知: {known_count} ({known_count/total_samples*100:.1f}%)")
        print(f"  全局阈值: {self.global_threshold:.6f}")
        print(f"  熵阈值: {entropy_threshold:.6f}")

        # 详细分析
        self._analyze_detection_results(features, predictions, min_distances, final_unknown_mask)

        return unknown_indices, known_indices

    def _analyze_detection_results(self, features: np.ndarray, predictions: np.ndarray,
                                 min_distances: np.ndarray, unknown_mask: np.ndarray):
        """
        分析检测结果
        """
        print("\n--- 检测结果详细分析 ---")

        # 分析预测分布
        pred_counts = defaultdict(int)
        unknown_pred_counts = defaultdict(int)

        for i, pred in enumerate(predictions):
            pred_counts[pred] += 1
            if unknown_mask[i]:
                unknown_pred_counts[pred] += 1

        print("预测类别分布:")
        for pred in sorted(pred_counts.keys()):
            total = pred_counts[pred]
            unknown = unknown_pred_counts.get(pred, 0)
            percentage = unknown / total * 100 if total > 0 else 0
            class_type = "已知" if pred <= self.known_class_count else "未知"
            print(f"  类别 {pred} ({class_type}): 总计 {total}, 未知 {unknown} ({percentage:.1f}%)")

        # 距离统计
        unknown_distances = min_distances[unknown_mask]
        known_distances = min_distances[~unknown_mask]

        if len(unknown_distances) > 0:
            print(f"\n被检测为未知的样本距离统计:")
            print(f"  平均距离: {np.mean(unknown_distances):.6f}")
            print(f"  距离范围: [{np.min(unknown_distances):.6f}, {np.max(unknown_distances):.6f}]")

        if len(known_distances) > 0:
            print(f"\n被检测为已知的样本距离统计:")
            print(f"  平均距离: {np.mean(known_distances):.6f}")
            print(f"  距离范围: [{np.min(known_distances):.6f}, {np.max(known_distances):.6f}]")

    def create_unknown_dataset(self, unlabeled_dataset: Subset, unknown_indices: List[int]) -> Subset:
        """
        创建未知类别数据集

        Args:
            unlabeled_dataset: 原始无标签数据集
            unknown_indices: 未知样本索引

        Returns:
            unknown_dataset: 只包含未知样本的数据集
        """
        print(f"创建未知类别数据集，包含 {len(unknown_indices)} 个样本")
        unknown_dataset = Subset(unlabeled_dataset, unknown_indices)
        return unknown_dataset

    def save_detection_results(self, unknown_indices: List[int], save_path: Path):
        """
        保存检测结果

        Args:
            unknown_indices: 未知样本索引列表
            save_path: 保存路径
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'unknown_indices': unknown_indices,
            'class_centroids': self.class_centroids,
            'global_threshold': self.global_threshold,
            'class_thresholds': self.class_thresholds,
            'known_class_count': self.known_class_count
        }

        torch.save(results, save_path)
        print(f"检测结果已保存到: {save_path}")