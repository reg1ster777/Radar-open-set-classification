"""绘图与可视化模块。

提供开集 UMAP 可视化、重构波形对比等图形输出能力。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch

import config


def plot_umap_open_set(embeddings, true_labels, cluster_labels, save_dir=None):
    """绘制 UMAP 嵌入，展示真实标签与聚类结果。"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors_known = plt.cm.tab10(np.linspace(0, 1, config.KNOWN_CLASS_COUNT))
    colors_unknown = plt.cm.Set3(np.linspace(0, 1, config.UNKNOWN_CLASS_COUNT))

    for i in range(config.KNOWN_CLASS_COUNT):
        mask = true_labels == i
        axes[0, 0].scatter(embeddings[mask, 0], embeddings[mask, 1],
                           c=[colors_known[i]], s=20, alpha=0.7, label=f'Known Class {i+1}')

    for i in range(config.UNKNOWN_CLASS_COUNT):
        mask = true_labels == (config.KNOWN_CLASS_COUNT + i)
        axes[0, 0].scatter(embeddings[mask, 0], embeddings[mask, 1],
                           c=[colors_unknown[i]], s=30, alpha=0.7, marker='^', label=f'Unknown Class {config.KNOWN_CLASS_COUNT + i + 1}')

    axes[0, 0].set_title('True Labels (Known vs Unknown)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('UMAP 1')
    axes[0, 0].set_ylabel('UMAP 2')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    unique_clusters = np.unique(cluster_labels)
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        if cluster_id == -1:
            mask = cluster_labels == cluster_id
            axes[0, 1].scatter(embeddings[mask, 0], embeddings[mask, 1],
                               c='black', s=20, alpha=0.5, label='Noise')
        else:
            mask = cluster_labels == cluster_id
            axes[0, 1].scatter(embeddings[mask, 0], embeddings[mask, 1],
                               c=[cluster_colors[i]], s=20, alpha=0.7,
                               label=f'Cluster {cluster_id}')

    axes[0, 1].set_title('KMeans Clustering Results', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('UMAP 1')
    axes[0, 1].set_ylabel('UMAP 2')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    known_mask = true_labels < config.KNOWN_CLASS_COUNT
    unknown_mask = ~known_mask
    axes[1, 0].scatter(embeddings[known_mask, 0], embeddings[known_mask, 1], c='blue', s=15, alpha=0.5, label='Known Classes')
    axes[1, 0].scatter(embeddings[unknown_mask, 0], embeddings[unknown_mask, 1], c='red', s=25, alpha=0.7, marker='^', label='Unknown Classes')
    axes[1, 0].set_title('Known vs Unknown Classes', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    axes[1, 0].legend()

    cluster_sizes = []
    cluster_names = []
    for cluster_id in unique_clusters:
        if cluster_id != -1:
            size = np.sum(cluster_labels == cluster_id)
            cluster_sizes.append(size)
            cluster_names.append(f'C{cluster_id}')
    axes[1, 1].bar(cluster_names, cluster_sizes)
    axes[1, 1].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Cluster ID')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "true_open_set_umap_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_reconstructions_per_class(
    class_id: int,
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    save_dir,
    samples_per_figure: int = 10,
) -> None:
    """绘制单个类别的原始 vs 重构波形拼图。"""
    assert original.shape == reconstructed.shape
    original_np = original.squeeze(1).cpu().numpy()
    reconstructed_np = reconstructed.squeeze(1).cpu().numpy()
    time_axis = np.arange(original_np.shape[1])

    rows, cols = 5, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows), sharex=True)
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx < min(samples_per_figure, original_np.shape[0]):
            ax.plot(time_axis, original_np[idx], label="Original", linewidth=1.2)
            ax.plot(
                time_axis,
                reconstructed_np[idx],
                label="Reconstruction",
                linewidth=1.0,
                linestyle="--",
            )
            ax.set_title(f"Class {class_id + 1} - Sample {idx + 1}")
            ax.grid(True, ls=":", alpha=0.4)
            if idx == 0:
                ax.legend(loc="upper right", fontsize=8)
        else:
            ax.axis("off")

        if idx // cols == rows - 1:
            ax.set_xlabel("Index")

    fig.suptitle(f"Class {class_id + 1} - First {samples_per_figure} Samples")
    fig.tight_layout()
    fig.savefig(save_dir / f"class_{class_id + 1:02d}_samples.png", dpi=300)
    plt.close(fig)
