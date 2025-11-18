import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Add this import
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Tuple
import numpy as np

from src.gcd_implementation.models import get_backbone, ReconstructionHead, LabeledReconstructionHead, ClassificationHead, get_labeled_reconstruction_head
from src.gcd_implementation.datasets import GCDDataset, mask_signal, augment_strong, augment_weak, get_gcd_split_indices


class GCDPretrainer:
    def __init__(
        self,
        arch_type: str,
        original_signal_length: int,
        input_channels: int,
        feature_dim: int,
        device: torch.device,
        lr: float = 1e-3,
        compression_ratio: float = 0.25,
    ):
        self.device = device

        self.backbone, _ = get_backbone(arch_type, input_channels=input_channels)
        self.reconstruction_head = ReconstructionHead(
            feature_dim=feature_dim,
            original_signal_length=original_signal_length,
            input_channels=input_channels,
            compression_ratio=compression_ratio
        ).to(device)

        self.backbone.to(device)
        self.optimizer = optim.Adam(
            list(self.backbone.parameters()) + list(self.reconstruction_head.parameters()),
            lr=lr
        )
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.backbone.train()  # 训练模式
        self.reconstruction_head.train()  # 重构模式
        
        total_loss = 0
        for batch_idx, (signals, _, _) in enumerate(tqdm(dataloader, desc="Pretraining")):
            signals = signals.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播：重构完整信号
            features = self.backbone(signals)
            reconstructed_signal, compressed_features = self.reconstruction_head(features)

            # 计算重构损失：重构完整信号 vs 原始信号
            loss = self.criterion(reconstructed_signal, signals)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def save_backbone(self, path: Path):
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'reconstruction_head_state_dict': self.reconstruction_head.state_dict(),
        }, path)
        print(f"Pre-trained backbone and reconstruction head saved to {path}")


class LabeledDataPretrainer:
    """有标签数据专用重构头预训练器"""
    def __init__(
        self,
        backbone: nn.Module,
        original_signal_length: int,
        input_channels: int,
        feature_dim: int,
        device: torch.device,
        lr: float = 1e-4,
    ):
        self.device = device
        self.backbone = backbone.to(device)

        # 创建有标签重构头
        self.labeled_reconstruction_head = get_labeled_reconstruction_head(
            feature_dim=feature_dim,
            original_signal_length=original_signal_length,
            input_channels=input_channels
        ).to(device)

        # 只训练重构头，冻结backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.labeled_reconstruction_head.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_epoch(self, labeled_dataloader: DataLoader) -> float:
        self.labeled_reconstruction_head.train()
        total_loss = 0

        for signals, labels, _ in tqdm(labeled_dataloader, desc="Labeled Reconstruction Pretraining"):
            signals = signals.to(self.device)

            self.optimizer.zero_grad()

            # 只在有标签数据上训练
            with torch.no_grad():
                features = self.backbone(signals)

            reconstructed_signal = self.labeled_reconstruction_head(features)
            loss = self.criterion(reconstructed_signal, signals)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(labeled_dataloader)

    def save_model(self, path: Path):
        torch.save({
            'labeled_reconstruction_head_state_dict': self.labeled_reconstruction_head.state_dict(),
        }, path)
        print(f"Labeled reconstruction head saved to {path}")


class GCDFinetuner:
    def __init__(
        self,
        backbone: nn.Module,
        labeled_reconstruction_head: nn.Module,
        feature_dim: int,
        num_known_classes: int,
        num_total_classes: int, # K + M
        device: torch.device,
        lr: float = 1e-3,
        freeze_backbone: bool = False, # 是否冻结backbone
        reconstruction_error_threshold: float = 0.1, # 重构误差阈值
    ):
        self.device = device
        self.backbone = backbone.to(device)
        self.labeled_reconstruction_head = labeled_reconstruction_head
        if self.labeled_reconstruction_head is not None:
            self.labeled_reconstruction_head = self.labeled_reconstruction_head.to(device)
        self.classification_head = ClassificationHead(
            feature_dim=feature_dim,
            num_classes=num_total_classes  # 保持10维输出，对应所有类别
        ).to(device)

        self.num_known_classes = num_known_classes
        self.num_total_classes = num_total_classes
        self.freeze_backbone = freeze_backbone
        self.reconstruction_error_threshold = reconstruction_error_threshold
        self.num_epochs = 0  # 将在训练时设置

        # 冻结backbone参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 重构头可能为None
            if self.labeled_reconstruction_head is not None:
                for param in self.labeled_reconstruction_head.parameters():
                    param.requires_grad = False
            print("✓ Backbone已冻结，只训练分类头")
            # 优化器只更新分类头
            self.optimizer = optim.Adam(self.classification_head.parameters(), lr=lr)
        else:
            print("✓ Backbone未冻结，联合训练backbone和分类头")
            # 优化器更新所有参数
            self.optimizer = optim.Adam(
                list(self.backbone.parameters()) + list(self.classification_head.parameters()),
                lr=lr
            )

        self.criterion_ce = nn.CrossEntropyLoss()

    def train_epoch(self, labeled_dataloader: DataLoader) -> float:
        self.backbone.train()
        self.classification_head.train()
        total_loss = 0

        for batch_idx, (labeled_signals, labeled_labels, _) in enumerate(tqdm(labeled_dataloader, desc="Supervised Fine-tuning")):
            labeled_signals = labeled_signals.to(self.device)
            labeled_labels = labeled_labels.to(self.device)

            self.optimizer.zero_grad()

            # ========== 有标签数据处理 ==========
            # 获取有标签特征
            features_labeled = self.backbone(labeled_signals)

            # 使用分类头进行分类
            logits_labeled = self.classification_head(features_labeled)
            # 将标签从1-based转换为0-based以适应CrossEntropyLoss
            labeled_labels_zero_based = labeled_labels - 1
            loss_labeled = self.criterion_ce(logits_labeled, labeled_labels_zero_based)

            # 反向传播和优化
            loss_labeled.backward()
            self.optimizer.step()
            total_loss += loss_labeled.item()

            # 输出训练信息
            if batch_idx == 0:
                print(f"  有标签损失: {loss_labeled.item():.4f}")

        return total_loss / len(labeled_dataloader)

    def save_model(self, path: Path):
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'classification_head_state_dict': self.classification_head.state_dict(),
        }, path)
        print(f"Fine-tuned model saved to {path}")


class UnknownClassFinetuner:
    """
    未知类别微调器
    专门用于在检测出的未知样本上进行微调，让模型更好地识别未知类别
    """
    def __init__(
        self,
        backbone: nn.Module,
        classification_head: nn.Module,
        known_class_count: int,
        num_total_classes: int,
        device: torch.device,
        lr: float = 1e-3,
        freeze_backbone: bool = False,
        entropy_weight: float = 0.1,  # 熵正则化权重
        distribution_sharpness: float = 5.0,  # 分布锐化参数
    ):
        self.device = device
        self.backbone = backbone.to(device)
        self.classification_head = classification_head.to(device)
        self.known_class_count = known_class_count
        self.num_total_classes = num_total_classes
        self.entropy_weight = entropy_weight
        self.distribution_sharpness = distribution_sharpness

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

        Args:
            logits: 模型输出logits [batch_size, num_classes]

        Returns:
            loss: 分布损失
        """
        # Mask已知类别
        masked_logits = logits.clone()
        masked_logits[:, :self.known_class_count] = -10.0  # 使用较大的负值

        # 获取每个样本在未知类别中的最大logits
        unknown_max_logits, unknown_max_indices = torch.max(masked_logits, dim=1)

        # 创建目标分布：每个样本只在检测到的最大未知类别上有高概率
        batch_size = logits.shape[0]
        target_distribution = torch.zeros_like(logits)

        # 使用锐化的softmax创建更集中的分布
        sharp_probs = F.softmax(unknown_max_logits * self.distribution_sharpness, dim=0)
        target_distribution.scatter_(1, unknown_max_indices.unsqueeze(1), sharp_probs.unsqueeze(1))

        # 计算当前概率分布
        current_probs = F.softmax(logits, dim=1)

        # 计算KL散度损失
        loss = F.kl_div(current_probs.log(), target_distribution, reduction='batchmean')

        return loss

    def entropy_regularization(self, logits: torch.Tensor) -> torch.Tensor:
        """
        熵正则化项，鼓励未知类别预测的多样性

        Args:
            logits: 模型输出logits [batch_size, num_classes]

        Returns:
            entropy_loss: 熵损失
        """
        # 只计算未知类别的熵
        masked_logits = logits.clone()
        masked_logits[:, :self.known_class_count] = -10.0

        unknown_probs = F.softmax(masked_logits, dim=1)
        entropy = -torch.sum(unknown_probs * torch.log(unknown_probs + 1e-8), dim=1)

        # 我们希望熵不要太小（避免过度集中）也不要太大（避免过于分散）
        # 使用二次函数，在目标熵值附近达到最小
        target_entropy = torch.log(torch.tensor(self.num_total_classes - self.known_class_count, dtype=torch.float32))
        entropy_loss = (entropy - target_entropy).pow(2).mean()

        return entropy_loss

    def train_epoch(self, unknown_dataloader: DataLoader) -> float:
        """
        训练一个epoch

        Args:
            unknown_dataloader: 未知样本数据加载器

        Returns:
            avg_loss: 平均损失
        """
        self.backbone.train()
        self.classification_head.train()
        total_loss = 0
        total_dist_loss = 0
        total_entropy_loss = 0

        for batch_idx, (signals, _, _) in enumerate(tqdm(unknown_dataloader, desc="Unknown Class Fine-tuning")):
            signals = signals.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            features = self.backbone(signals)
            logits = self.classification_head(features)

            # 计算未知类别分布损失
            dist_loss = self.unknown_distribution_loss(logits)

            # 计算熵正则化损失
            entropy_loss = self.entropy_regularization(logits)

            # 总损失
            total_loss_batch = dist_loss + self.entropy_weight * entropy_loss

            # 反向传播
            total_loss_batch.backward()
            self.optimizer.step()

            total_loss += total_loss_batch.item()
            total_dist_loss += dist_loss.item()
            total_entropy_loss += entropy_loss.item()

            # 输出训练信息
            if batch_idx == 0:
                print(f"  损失分布: 分布损失 {dist_loss.item():.4f}, 熵损失 {entropy_loss.item():.4f}")

        avg_loss = total_loss / len(unknown_dataloader)
        avg_dist_loss = total_dist_loss / len(unknown_dataloader)
        avg_entropy_loss = total_entropy_loss / len(unknown_dataloader)

        print(f"  平均损失: 总损失 {avg_loss:.4f}, 分布损失 {avg_dist_loss:.4f}, 熵损失 {avg_entropy_loss:.4f}")

        return avg_loss

    def evaluate_unknown_predictions(self, unknown_dataloader: DataLoader) -> dict:
        """
        评估未知样本的预测质量

        Args:
            unknown_dataloader: 未知样本数据加载器

        Returns:
            metrics: 评估指标
        """
        self.backbone.eval()
        self.classification_head.eval()

        all_logits = []
        all_predictions = []

        with torch.no_grad():
            for signals, _, _ in unknown_dataloader:
                signals = signals.to(self.device)

                features = self.backbone(signals)
                logits = self.classification_head(features)

                # 获取预测（mask已知类别后）
                masked_logits = logits.clone()
                masked_logits[:, :self.known_class_count] = -10.0

                predictions = torch.argmax(masked_logits, dim=1) + self.known_class_count + 1  # 转换为1-based

                all_logits.append(logits.cpu())
                all_predictions.append(predictions.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)

        # 计算预测分布
        pred_dist = {}
        for pred in all_predictions.tolist():
            pred_dist[pred] = pred_dist.get(pred, 0) + 1

        total_samples = len(all_predictions)
        pred_percentages = {k: v/total_samples*100 for k, v in pred_dist.items()}

        metrics = {
            'total_samples': total_samples,
            'prediction_distribution': pred_percentages,
            'avg_confidence': torch.max(F.softmax(all_logits, dim=1), dim=1)[0].mean().item(),
            'avg_entropy': -torch.sum(F.softmax(all_logits, dim=1) * torch.log(F.softmax(all_logits, dim=1) + 1e-8), dim=1).mean().item()
        }

        return metrics

    def save_model(self, path: Path):
        """
        保存微调后的模型

        Args:
            path: 保存路径
        """
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'classification_head_state_dict': self.classification_head.state_dict(),
            'known_class_count': self.known_class_count,
            'num_total_classes': self.num_total_classes,
            'entropy_weight': self.entropy_weight,
            'distribution_sharpness': self.distribution_sharpness,
        }, path)
        print(f"Unknown class fine-tuned model saved to {path}")


class GCDEvaluator:
    def __init__(
        self,
        backbone: nn.Module,
        classification_head: nn.Module,
        device: torch.device,
    ):
        self.backbone = backbone.to(device)
        self.classification_head = classification_head.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, num_total_classes: int) -> Tuple[float, np.ndarray]:
        self.backbone.eval()
        self.classification_head.eval()

        all_preds = []
        all_labels = []

        for signals, labels, _ in tqdm(dataloader, desc="Evaluating"):
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            features = self.backbone(signals)
            logits = self.classification_head(features)
            predictions = torch.argmax(logits, dim=1)

            # 将预测结果从0-based转换回1-based以匹配标签范围
            predictions = predictions + 1

            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # --- 打印每个真实标签下的分类结果占比 ---
        print("\n--- 详细分类结果占比 (每个真实标签下) ---")
        unique_true_labels = np.unique(all_labels)
        for true_label in sorted(unique_true_labels):
            # 筛选出真实标签为 current_true_label 的所有样本的预测结果
            preds_for_this_label = all_preds[all_labels == true_label]
            
            if len(preds_for_this_label) == 0:
                print(f"真实标签 {true_label}: 无样本")
                continue

            # 计算这些预测结果的分布
            unique_preds, counts = np.unique(preds_for_this_label, return_counts=True)
            proportions = counts / len(preds_for_this_label)
            
            # 格式化输出
            distribution_str = []
            for pred_label, prop in zip(unique_preds, proportions):
                distribution_str.append(f"簇 {pred_label} ({prop:.2%})")
            print(f"真实标签 {true_label}: {', '.join(distribution_str)}")
        print("-----------------------------------------")

        # --- GCD 评估：不再进行匈牙利算法匹配，只打印详细分布 ---
        # 返回一个占位符准确率，因为用户不再需要匹配后的准确率
        placeholder_accuracy = 0.0
        print(f"Evaluation Accuracy (Hungarian matching skipped): {placeholder_accuracy:.4f}")
        return placeholder_accuracy, all_preds
