"""监督分类训练模块。

在预训练完成后，对已知类别进行交叉熵微调以及准确率评估。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data_utils import build_supervised_loader


@torch.no_grad()
def evaluate_classifier(backbone: nn.Module, classifier: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """计算分类器在指定数据集上的准确率。"""
    backbone.eval()
    classifier.eval()
    correct = 0
    total = 0
    for signals, labels in dataloader:
        signals = signals.to(device)
        labels = labels.to(device)
        logits = classifier(backbone(signals))
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_supervised_classifier(
    pretrainer,
    train_signals: torch.Tensor,
    train_labels: torch.Tensor,
    val_signals: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    num_classes: int,
    *,
    batch_size: int = config.SUPERVISED_BATCH_SIZE,
    epochs: int = config.SUPERVISED_EPOCHS,
    lr: float = config.SUPERVISED_LR,
    freeze_backbone: bool = config.FREEZE_BACKBONE_IN_SUP,
) -> Tuple[nn.Module, dict]:
    """在已知类上训练一个线性分类头。"""
    backbone = pretrainer.backbone
    train_loader = build_supervised_loader(train_signals, train_labels, batch_size=batch_size, shuffle=True)
    val_loader = build_supervised_loader(val_signals, val_labels, batch_size=batch_size, shuffle=False)

    feature_dim = backbone(train_signals[:1].to(device)).shape[1]
    classifier = nn.Linear(feature_dim, num_classes).to(device)

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(
            list(backbone.parameters()) + list(classifier.parameters()), lr=lr
        )

    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_acc": []}
    best_state = None
    best_val_acc = 0.0

    for epoch in range(epochs):
        backbone.train()
        classifier.train()
        running_loss = 0.0
        steps = 0
        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            logits = classifier(backbone(signals))
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

        avg_loss = running_loss / max(steps, 1)
        val_acc = evaluate_classifier(backbone, classifier, val_loader, device)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = classifier.state_dict()

        print(f"[Supervised] Epoch {epoch + 1}/{epochs} - loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    if best_state is not None:
        classifier.load_state_dict(best_state)
    history["best_val_acc"] = best_val_acc
    return classifier, history
