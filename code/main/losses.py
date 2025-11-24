"""损失函数模块。

包含监督式对比学习损失和基于类原型的交叉熵损失，用于在预训练阶段注入标签监督。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """有标签对比损失：同类样本互为正样本。"""
    device = features.device
    batch_size = features.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=device)

    features = F.normalize(features, dim=1)
    similarity = torch.matmul(features, features.T) / temperature
    similarity = similarity - torch.max(similarity, dim=1, keepdim=True)[0].detach()

    logits_mask = torch.ones_like(similarity, device=device) - torch.eye(batch_size, device=device)
    similarity = similarity * logits_mask + (-1e9) * (1 - logits_mask)

    exp_logits = torch.exp(similarity)
    log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    labels = labels.contiguous().view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(device)
    positive_mask = positive_mask * logits_mask

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_count + 1e-12)
    loss = -mean_log_prob_pos[valid].mean()
    return loss


def prototype_cross_entropy(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """使用 batch 内类原型的交叉熵损失，不额外引入分类头。"""
    device = features.device
    unique_labels, mapped = torch.unique(labels, sorted=True, return_inverse=True)
    class_count = unique_labels.size(0)
    if class_count <= 1:
        return torch.tensor(0.0, device=device)

    prototypes = torch.zeros(class_count, features.size(1), device=device)
    prototypes.index_add_(0, mapped, features)
    counts = torch.zeros(class_count, device=device)
    counts.index_add_(0, mapped, torch.ones_like(mapped, dtype=torch.float32))
    prototypes = prototypes / counts.unsqueeze(1)

    logits = features @ prototypes.T
    ce_loss = F.cross_entropy(logits, mapped)
    return ce_loss
