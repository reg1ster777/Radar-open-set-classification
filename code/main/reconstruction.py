"""重构式预训练与波形可视化模块。

封装 backbone+重构头的联合训练逻辑（含监督式对比/交叉熵），以及重构结果的可视化。
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.gcd_implementation.gcd_solver import GCDPretrainer
import config
from data_utils import ensure_channel_dim
from losses import supervised_contrastive_loss, prototype_cross_entropy
from plotting import plot_reconstructions_per_class


def train_feature_extractor(
    signals: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
) -> GCDPretrainer:
    """在重构自监督基础上引入监督对比 + 原型交叉熵联合训练。"""
    signals = ensure_channel_dim(signals)
    signal_length = signals.shape[-1]
    channels = signals.shape[1]
    pretrainer = GCDPretrainer(
        arch_type="ca1d",
        original_signal_length=signal_length,
        input_channels=channels,
        feature_dim=128,
        device=device,
        lr=1e-4,
        compression_ratio=0.25,
    )

    dataset = TensorDataset(signals, labels)
    dataloader = DataLoader(dataset, batch_size=config.RECON_BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        pretrainer.backbone.train()
        pretrainer.reconstruction_head.train()

        recon_running = 0.0
        contrast_running = 0.0
        ce_running = 0.0
        steps = 0

        for batch_signals, batch_labels in dataloader:
            batch_signals = batch_signals.to(device)
            batch_labels = batch_labels.to(device)

            pretrainer.optimizer.zero_grad()

            features = pretrainer.backbone(batch_signals)
            reconstructed_signal, _ = pretrainer.reconstruction_head(features)
            recon_loss = pretrainer.criterion(reconstructed_signal, batch_signals)

            sup_contrast_loss = supervised_contrastive_loss(
                features, batch_labels, config.SUPERVISED_CONTRASTIVE_TEMPERATURE
            )
            ce_loss = prototype_cross_entropy(features, batch_labels)

            loss = (
                recon_loss
                + config.SUPERVISED_CONTRASTIVE_WEIGHT * sup_contrast_loss
                + config.SUPERVISED_CE_WEIGHT * ce_loss
            )
            loss.backward()
            pretrainer.optimizer.step()

            recon_running += recon_loss.item()
            contrast_running += sup_contrast_loss.item()
            ce_running += ce_loss.item()
            steps += 1

        recon_avg = recon_running / max(steps, 1)
        contrast_avg = contrast_running / max(steps, 1)
        ce_avg = ce_running / max(steps, 1)
        print(f"Epoch {epoch + 1}/{epochs} - Recon={recon_avg:.6f}  SupCon={contrast_avg:.6f}  CE={ce_avg:.6f}")
    return pretrainer


def reconstruct_samples(pretrainer: GCDPretrainer, samples: torch.Tensor, device: torch.device) -> torch.Tensor:
    """利用训练好的重构器生成波形重构。"""
    pretrainer.backbone.eval()
    pretrainer.reconstruction_head.eval()
    with torch.no_grad():
        signals = samples.to(device)
        features = pretrainer.backbone(signals)
        reconstructed, _ = pretrainer.reconstruction_head(features)
    return reconstructed.cpu()


def visualize_reconstruction_per_class(
    train_signals: torch.Tensor,
    train_labels: torch.Tensor,
    pretrainer: GCDPretrainer,
    device: torch.device,
    save_dir,
    samples_per_class: int = 10,
) -> None:
    """遍历每个已知类，绘制若干样本的原始 vs 重构对比。"""
    print("\nReconstructing first 10 samples per class...")
    train_signals = ensure_channel_dim(train_signals)
    for class_id in range(config.KNOWN_CLASS_COUNT):
        indices = torch.nonzero(train_labels == class_id, as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            print(f"Class {class_id + 1}: no samples found, skipping.")
            continue
        selected = indices[:samples_per_class]
        class_original = train_signals[selected].clone()
        reconstructed = reconstruct_samples(pretrainer, class_original, device)
        mse = torch.mean((reconstructed - class_original) ** 2).item()
        print(f"Class {class_id + 1}: MSE on first {len(selected)} samples = {mse:.6f}")
        plot_reconstructions_per_class(
            class_id,
            class_original,
            reconstructed,
            save_dir=save_dir,
            samples_per_figure=samples_per_class,
        )
    print(f"Visualization saved to: {save_dir}")
