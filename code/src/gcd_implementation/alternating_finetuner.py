import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Tuple, List
import numpy as np

from src.gcd_implementation.models import ClassificationHead


class AlternatingFinetuner:
    """
    穿插式微调器
    在同一个epoch中同时处理有标签和未知类别数据
    实现平衡的已知/未知类别学习
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_known_classes: int,
        num_total_classes: int,
        device: torch.device,
        lr: float = 1e-3,
        freeze_backbone: bool = False,
        labeled_weight: float = 1.0,  # 有标签损失权重
        unknown_weight: float = 0.5,  # 未知损失权重
        unknown_ratio: float = 0.3,  # 每个batch中未知样本的比例
        distribution_sharpness: float = 5.0,  # 未知类别分布锐度
        entropy_weight: float = 0.1,  # 熵正则化权重
    ):
        self.device = device
        self.backbone = backbone.to(device)
        self.num_known_classes = num_known_classes
        self.num_total_classes = num_total_classes
        self.labeled_weight = labeled_weight
        self.unknown_weight = unknown_weight
        self.unknown_ratio = unknown_ratio
        self.distribution_sharpness = distribution_sharpness
        self.entropy_weight = entropy_weight

        # 创建分类头
        self.classification_head = ClassificationHead(
            feature_dim=feature_dim,
            num_classes=num_total_classes
        ).to(device)

        # 冻结backbone参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Backbone已冻结，只训练分类头")
            self.optimizer = optim.Adam(self.classification_head.parameters(), lr=lr)
        else:
            print("✓ Backbone未冻结，联合训练backbone和分类头")
            self.optimizer = optim.Adam(
                list(self.backbone.parameters()) + list(self.classification_head.parameters()),
                lr=lr
            )

        self.criterion_ce = nn.CrossEntropyLoss()

    def unknown_distribution_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算未知类别分布损失
        让logits分布集中到mask后最大logits对应的未知类别
        """
        # Mask已知类别 - 使用更强的mask确保已知类别概率接近0
        masked_logits = logits.clone()
        masked_logits[:, :self.num_known_classes] = -100.0

        # 获取每个样本在未知类别中的最大logits
        unknown_max_logits, unknown_max_indices = torch.max(masked_logits, dim=1)

        # 创建目标分布：每个样本集中在其最可能的未知类别上
        batch_size = logits.shape[0]
        target_distribution = torch.zeros_like(logits)

        # 对于未知样本分布损失，我们希望每个样本都集中在其预测的未知类别上
        # 这里不需要在batch维度上进行softmax，每个样本应该是独立的
        # 使用distribution_sharpness来控制每个样本分布的集中程度
        confidence = torch.sigmoid(unknown_max_logits * self.distribution_sharpness)

        # 每个样本在其预测的未知类别上分配高置信度
        target_distribution.scatter_(1, unknown_max_indices.unsqueeze(1), confidence.unsqueeze(1))

        # 计算当前概率分布
        current_probs = F.softmax(logits, dim=1)

        # 计算KL散度损失
        loss = F.kl_div(current_probs.log(), target_distribution, reduction='batchmean')

        return loss

    def entropy_regularization(self, logits: torch.Tensor) -> torch.Tensor:
        """
        熵正则化项，鼓励未知类别预测的多样性
        """
        # 只计算未知类别的熵 - 使用更强的mask
        masked_logits = logits.clone()
        masked_logits[:, :self.num_known_classes] = -100.0

        unknown_probs = F.softmax(masked_logits, dim=1)
        entropy = -torch.sum(unknown_probs * torch.log(unknown_probs + 1e-8), dim=1)

        # 我们希望熵不要太小（避免过度集中）也不要太大（避免过于分散）
        target_entropy = torch.log(torch.tensor(self.num_total_classes - self.num_known_classes, dtype=torch.float32))
        entropy_loss = (entropy - target_entropy).pow(2).mean()

        return entropy_loss

    def create_mixed_batch(self, labeled_batch, unknown_batch):
        """
        创建混合批次，包含有标签和未知样本

        Args:
            labeled_batch: (signals, labels, original_labels)
            unknown_batch: (signals, labels, original_labels)

        Returns:
            mixed_signals: 混合信号
            mixed_labels: 混合标签（用于有标签监督学习）
            sample_types: 样本类型（0=有标签，1=未知）
        """
        labeled_signals, labeled_labels, _ = labeled_batch
        unknown_signals, unknown_labels, _ = unknown_batch

        labeled_size = labeled_signals.shape[0]
        unknown_size = unknown_signals.shape[0]

        # 混合信号 - 确保所有张量都在同一个设备上
        labeled_signals = labeled_signals.to(self.device)
        unknown_signals = unknown_signals.to(self.device)
        mixed_signals = torch.cat([labeled_signals, unknown_signals], dim=0)

        # 创建样本类型标记
        sample_types = torch.cat([
            torch.zeros(labeled_size, dtype=torch.float32, device=self.device),  # 0 = 有标签
            torch.ones(unknown_size, dtype=torch.float32, device=self.device)    # 1 = 未知
        ], dim=0)

        # 混合标签（有标签样本使用真实标签，未知样本使用-1作为占位符）
        # 确保所有张量都在同一个设备上
        labeled_labels = labeled_labels.to(self.device)
        unknown_labels = torch.full((unknown_size,), -1, dtype=torch.long, device=self.device)
        mixed_labels = torch.cat([labeled_labels, unknown_labels], dim=0)

        return mixed_signals, mixed_labels, sample_types

    def train_epoch_alternating(self, labeled_dataloader: DataLoader, unknown_dataloader: DataLoader) -> dict:
        """
        穿插式训练一个epoch
        在每个step中同时处理有标签和未知样本

        Args:
            labeled_dataloader: 有标签数据加载器
            unknown_dataloader: 未知类别数据加载器

        Returns:
            metrics: 训练指标
        """
        self.backbone.train()
        self.classification_head.train()

        total_labeled_loss = 0
        total_unknown_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        num_labeled_samples = 0
        num_unknown_samples = 0

        # 创建未知数据迭代器
        unknown_iter = iter(unknown_dataloader)

        for batch_idx, labeled_batch in enumerate(tqdm(labeled_dataloader, desc="Alternating Fine-tuning")):
            try:
                unknown_batch = next(unknown_iter)
            except StopIteration:
                unknown_iter = iter(unknown_dataloader)
                unknown_batch = next(unknown_iter)

            # 创建混合批次
            mixed_signals, mixed_labels, sample_types = self.create_mixed_batch(labeled_batch, unknown_batch)

            mixed_signals = mixed_signals.to(self.device)
            mixed_labels = mixed_labels.to(self.device)
            sample_types = sample_types.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            features = self.backbone(mixed_signals)
            logits = self.classification_head(features)

            # 分离有标签和未知样本
            labeled_mask = sample_types == 0
            unknown_mask = sample_types == 1

            # 有标签样本的监督损失
            labeled_loss = torch.tensor(0.0, device=self.device)
            if labeled_mask.any():
                labeled_logits = logits[labeled_mask]
                labeled_true_labels = mixed_labels[labeled_mask]

                # 将标签从1-based转换为0-based
                labeled_true_labels_zero_based = labeled_true_labels - 1
                labeled_loss = self.criterion_ce(labeled_logits, labeled_true_labels_zero_based)

                num_labeled_samples += labeled_mask.sum().item()

            # 未知样本的分布损失
            unknown_loss = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)
            if unknown_mask.any():
                unknown_logits = logits[unknown_mask]
                unknown_loss = self.unknown_distribution_loss(unknown_logits)
                entropy_loss = self.entropy_regularization(unknown_logits)

                num_unknown_samples += unknown_mask.sum().item()

            # 组合损失
            total_loss_batch = (
                self.labeled_weight * labeled_loss +
                self.unknown_weight * (unknown_loss + self.entropy_weight * entropy_loss)
            )

            # 反向传播
            total_loss_batch.backward()
            self.optimizer.step()

            # 累计损失
            total_labeled_loss += labeled_loss.item()
            total_unknown_loss += unknown_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_loss += total_loss_batch.item()

            # 输出训练信息
            if batch_idx == 0:
                print(f"  损失分布: 有标签 {labeled_loss.item():.4f}, 未知分布 {unknown_loss.item():.4f}, 熵 {entropy_loss.item():.4f}")
                print(f"  样本比例: 有标签 {num_labeled_samples}, 未知 {num_unknown_samples}")

        # 计算平均损失
        num_batches = len(labeled_dataloader)
        avg_metrics = {
            'total_loss': total_loss / num_batches,
            'labeled_loss': total_labeled_loss / num_batches,
            'unknown_loss': total_unknown_loss / num_batches,
            'entropy_loss': total_entropy_loss / num_batches,
            'num_labeled_samples': num_labeled_samples,
            'num_unknown_samples': num_unknown_samples
        }

        print(f"平均损失: 总损失 {avg_metrics['total_loss']:.4f}")
        print(f"  有标签损失: {avg_metrics['labeled_loss']:.4f}")
        print(f"  未知损失: {avg_metrics['unknown_loss']:.4f}")
        print(f"  熵损失: {avg_metrics['entropy_loss']:.4f}")

        return avg_metrics

    def evaluate_alternating(self, labeled_dataloader: DataLoader, unknown_dataloader: DataLoader) -> dict:
        """
        评估穿插式微调的效果

        Args:
            labeled_dataloader: 有标签数据加载器
            unknown_dataloader: 未知类别数据加载器

        Returns:
            metrics: 评估指标
        """
        self.backbone.eval()
        self.classification_head.eval()

        # 评估有标签数据
        labeled_correct = 0
        labeled_total = 0
        labeled_predictions = []
        labeled_true_labels = []

        with torch.no_grad():
            for signals, labels, _ in labeled_dataloader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)

                features = self.backbone(signals)
                logits = self.classification_head(features)
                predictions = torch.argmax(logits, dim=1) + 1  # 转换为1-based

                labeled_correct += (predictions == labels).sum().item()
                labeled_total += labels.size(0)

                labeled_predictions.extend(predictions.cpu().numpy())
                labeled_true_labels.extend(labels.cpu().numpy())

        # 评估未知类别数据
        unknown_predictions = []
        unknown_distributions = []

        with torch.no_grad():
            for signals, _, _ in unknown_dataloader:
                signals = signals.to(self.device)

                features = self.backbone(signals)
                logits = self.classification_head(features)

                # 获取预测分布
                probs = F.softmax(logits, dim=1)

                # Mask已知类别，看未知类别的分布 - 使用更强的mask
                masked_logits = logits.clone()
                masked_logits[:, :self.num_known_classes] = -100.0
                unknown_probs = F.softmax(masked_logits, dim=1)

                predictions = torch.argmax(masked_logits, dim=1) + self.num_known_classes + 1

                unknown_predictions.extend(predictions.cpu().numpy())
                unknown_distributions.extend(unknown_probs.cpu().numpy())

        # 计算指标
        labeled_accuracy = labeled_correct / labeled_total if labeled_total > 0 else 0

        # 未知类别预测分布统计
        unknown_pred_counts = {}
        for pred in unknown_predictions:
            unknown_pred_counts[pred] = unknown_pred_counts.get(pred, 0) + 1

        metrics = {
            'labeled_accuracy': labeled_accuracy,
            'labeled_total': labeled_total,
            'unknown_total': len(unknown_predictions),
            'unknown_distribution': unknown_pred_counts,
            'avg_confidence': np.mean([np.max(dist) for dist in unknown_distributions]) if unknown_distributions else 0
        }

        print(f"\n评估结果:")
        print(f"  有标签准确率: {labeled_accuracy:.4f} ({labeled_correct}/{labeled_total})")
        print(f"  未知样本数: {metrics['unknown_total']}")
        print(f"  未知类别预测分布: {metrics['unknown_distribution']}")
        print(f"  平均置信度: {metrics['avg_confidence']:.4f}")

        return metrics

    def save_model(self, path):
        """
        保存模型

        Args:
            path: 保存路径
        """
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'classification_head_state_dict': self.classification_head.state_dict(),
            'num_known_classes': self.num_known_classes,
            'num_total_classes': self.num_total_classes,
            'labeled_weight': self.labeled_weight,
            'unknown_weight': self.unknown_weight,
            'unknown_ratio': self.unknown_ratio,
            'distribution_sharpness': self.distribution_sharpness,
            'entropy_weight': self.entropy_weight,
        }, path)
        print(f"Alternating fine-tuned model saved to {path}")