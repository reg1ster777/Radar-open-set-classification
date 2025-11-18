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
    """
    未知类别检测器
    用于从无标签数据中分离出未知类别的样本
    """

    def __init__(self, backbone, classification_head, device, known_class_count=7):
        """
        初始化未知类别检测器

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

    def predict_with_confidence(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测并获取置信度分数

        Args:
            dataloader: 无标签数据加载器

        Returns:
            predictions: 预测结果 [N]
            confidences: 置信度分数 [N]
            logits: 原始logits [N, num_classes]
        """
        all_predictions = []
        all_confidences = []
        all_logits = []

        with torch.no_grad():
            for signals, _, _ in dataloader:
                signals = signals.to(self.device)

                # 前向传播
                features = self.backbone(signals)
                logits = self.classification_head(features)

                # 获取预测和置信度
                probs = F.softmax(logits, dim=1)
                predictions = torch.argmax(probs, dim=1)
                confidences = torch.max(probs, dim=1)[0]

                # 转换回1-based标签
                predictions = predictions + 1

                all_predictions.append(predictions.cpu())
                all_confidences.append(confidences.cpu())
                all_logits.append(logits.cpu())

        predictions = torch.cat(all_predictions, dim=0)
        confidences = torch.cat(all_confidences, dim=0)
        logits = torch.cat(all_logits, dim=0)

        return predictions, confidences, logits

    def detect_unknown_samples(self, unlabeled_dataloader: DataLoader,
                              confidence_threshold: float = 0.6,
                              use_entropy_threshold: bool = True,
                              entropy_threshold: float = 1.5,
                              use_class_balance: bool = True,
                              unknown_ratio: float = 0.3) -> Tuple[List[int], List[int]]:
        """
        检测未知样本

        Args:
            unlabeled_dataloader: 无标签数据加载器
            confidence_threshold: 置信度阈值
            use_entropy_threshold: 是否使用熵阈值
            entropy_threshold: 熵阈值

        Returns:
            unknown_indices: 未知样本的索引列表
            known_indices: 已知样本的索引列表
        """
        print("检测未知样本...")
        predictions, confidences, logits = self.predict_with_confidence(unlabeled_dataloader)

        # 计算熵
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        # 方法1: 基于置信度的检测
        low_confidence_mask = confidences < confidence_threshold

        # 方法2: 基于熵的检测（可选）
        if use_entropy_threshold:
            high_entropy_mask = entropy > entropy_threshold
            unknown_mask = low_confidence_mask | high_entropy_mask
        else:
            unknown_mask = low_confidence_mask

        # 方法3: 基于预测类别的检测（如果预测为已知类别但置信度低）
        predicted_known_classes = predictions <= self.known_class_count
        low_confidence_known = predicted_known_classes & low_confidence_mask

        # 最终未知样本判断
        final_unknown_mask = unknown_mask

        # 新增：如果所有样本都被预测为已知类别，强制采样未知样本
        if use_class_balance and torch.sum(predictions > self.known_class_count) == 0:
            print("警告：模型没有预测任何未知类别，使用强制采样策略")

            # 策略1：选择置信度最低的样本作为未知样本
            num_unknown = int(len(predictions) * unknown_ratio)
            _, low_conf_indices = torch.topk(confidences, len(predictions), largest=False)
            forced_unknown_indices = low_conf_indices[:num_unknown]

            final_unknown_mask = torch.zeros_like(final_unknown_mask, dtype=torch.bool)
            final_unknown_mask[forced_unknown_indices] = True

            # 策略2：结合熵高的样本
            high_entropy_mask = entropy > torch.quantile(entropy, 1 - unknown_ratio)
            final_unknown_mask = final_unknown_mask | high_entropy_mask

        # 收集索引
        unknown_indices = []
        known_indices = []

        batch_size = unlabeled_dataloader.batch_size
        for i in range(len(predictions)):
            global_idx = i  # 由于DataLoader的shuffle=False，索引是连续的
            if final_unknown_mask[i]:
                unknown_indices.append(global_idx)
            else:
                known_indices.append(global_idx)

        # 输出统计信息
        total_samples = len(predictions)
        unknown_count = len(unknown_indices)
        known_count = len(known_indices)

        print(f"未知样本检测结果:")
        print(f"  总样本数: {total_samples}")
        print(f"  检测为未知: {unknown_count} ({unknown_count/total_samples*100:.1f}%)")
        print(f"  检测为已知: {known_count} ({known_count/total_samples*100:.1f}%)")
        print(f"  置信度阈值: {confidence_threshold}")
        if use_entropy_threshold:
            print(f"  熵阈值: {entropy_threshold}")

        # 按原始标签分析检测效果（如果有的话）
        self._analyze_detection_quality(predictions, confidences, entropy, final_unknown_mask)

        return unknown_indices, known_indices

    def _analyze_detection_quality(self, predictions: torch.Tensor, confidences: torch.Tensor,
                                 entropy: torch.Tensor, unknown_mask: torch.Tensor):
        """
        分析检测质量（仅用于调试，实际应用中没有真实标签）
        """
        print("\n--- 检测质量分析 ---")

        # 分析被检测为未知的样本
        unknown_predictions = predictions[unknown_mask]
        unknown_confidences = confidences[unknown_mask]

        if len(unknown_predictions) > 0:
            print(f"被检测为未知的样本分析:")
            print(f"  预测为已知类别(1-{self.known_class_count}): {(unknown_predictions <= self.known_class_count).sum().item()} 个")
            print(f"  预测为未知类别({self.known_class_count+1}+): {(unknown_predictions > self.known_class_count).sum().item()} 个")
            print(f"  平均置信度: {unknown_confidences.mean().item():.4f}")
            print(f"  平均熵: {entropy[unknown_mask].mean().item():.4f}")

        # 分析被检测为已知的样本
        known_mask = ~unknown_mask
        known_predictions = predictions[known_mask]
        known_confidences = confidences[known_mask]

        if len(known_predictions) > 0:
            print(f"\n被检测为已知的样本分析:")
            print(f"  预测为已知类别(1-{self.known_class_count}): {(known_predictions <= self.known_class_count).sum().item()} 个")
            print(f"  预测为未知类别({self.known_class_count+1}+): {(known_predictions > self.known_class_count).sum().item()} 个")
            print(f"  平均置信度: {known_confidences.mean().item():.4f}")
            print(f"  平均熵: {entropy[known_mask].mean().item():.4f}")

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

    def save_unknown_indices(self, unknown_indices: List[int], save_path: Path):
        """
        保存未知样本索引

        Args:
            unknown_indices: 未知样本索引列表
            save_path: 保存路径
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(unknown_indices, save_path)
        print(f"未知样本索引已保存到: {save_path}")

    def load_unknown_indices(self, load_path: Path) -> List[int]:
        """
        加载未知样本索引

        Args:
            load_path: 加载路径

        Returns:
            unknown_indices: 未知样本索引列表
        """
        if load_path.exists():
            unknown_indices = torch.load(load_path)
            print(f"从 {load_path} 加载了 {len(unknown_indices)} 个未知样本索引")
            return unknown_indices
        else:
            print(f"未找到 {load_path}，需要重新检测未知样本")
            return None