#!/usr/bin/env python3
"""Open-set feature extraction, reconstruction, and visualization for data1."""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gcd_implementation.gcd_solver import GCDPretrainer
from src.solver.dataset import load_radar_known_fixed_test, load_radar_unknown

DATA_ROOT = "data1\data_noise_50"
KNOWN_CLASS_COUNT = 7
UNKNOWN_CLASS_COUNT = 3
TOTAL_CLASSES = KNOWN_CLASS_COUNT + UNKNOWN_CLASS_COUNT

RECON_DIR = Path(__file__).resolve().parent / "reconstruction_outputs_50"
RECON_DIR.mkdir(parents=True, exist_ok=True)
OPEN_SET_DIR = Path(__file__).resolve().parent / "true_open_set_results_data1_50"
OPEN_SET_DIR.mkdir(parents=True, exist_ok=True)


def plot_umap_open_set(embeddings, true_labels, cluster_labels, save_dir=None):
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors_known = plt.cm.tab10(np.linspace(0, 1, KNOWN_CLASS_COUNT))
    colors_unknown = plt.cm.Set3(np.linspace(0, 1, UNKNOWN_CLASS_COUNT))

    for i in range(KNOWN_CLASS_COUNT):
        mask = true_labels == i
        axes[0, 0].scatter(embeddings[mask, 0], embeddings[mask, 1],
                           c=[colors_known[i]], s=20, alpha=0.7, label=f'Known Class {i+1}')

    for i in range(UNKNOWN_CLASS_COUNT):
        mask = true_labels == (KNOWN_CLASS_COUNT + i)
        axes[0, 0].scatter(embeddings[mask, 0], embeddings[mask, 1],
                           c=[colors_unknown[i]], s=30, alpha=0.7, marker='^', label=f'Unknown Class {KNOWN_CLASS_COUNT + i + 1}')

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

    axes[0, 1].set_title('DBSCAN Clustering Results', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('UMAP 1')
    axes[0, 1].set_ylabel('UMAP 2')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    known_mask = true_labels < KNOWN_CLASS_COUNT
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


def create_open_set_dataset():
    print("创建开集数据集 (data1)...")
    train_loader_known, test_loader_known = load_radar_known_fixed_test(
        root=str(DATA_ROOT),
        known_class_count=KNOWN_CLASS_COUNT,
        test_per_class=1000,
        batch_size=256,
        shuffle_train=False,
        shuffle_test=False,
        as_loader=True,
    )

    unknown_loader = load_radar_unknown(
        root=str(DATA_ROOT),
        known_class_count=KNOWN_CLASS_COUNT,
        batch_size=256,
        as_loader=True,
    )

    train_batches = [(data, labels) for data, labels in train_loader_known]
    test_batches = [(data, labels) for data, labels in test_loader_known]
    unknown_batches = [(data, labels) for data, labels in unknown_loader]

    train_known_data = torch.cat([data for data, _ in train_batches], dim=0)
    train_known_labels = torch.cat([labels for _, labels in train_batches], dim=0)

    test_known_data = torch.cat([data for data, _ in test_batches], dim=0)
    test_known_labels = torch.cat([labels for _, labels in test_batches], dim=0)

    unknown_data = torch.cat([data for data, _ in unknown_batches], dim=0)

    print(f"  已知类训练样本: {len(train_known_data)}")
    print(f"  已知类测试样本: {len(test_known_data)}")
    print(f"  未知类样本总数: {len(unknown_data)}")

    samples_per_unknown_class = len(unknown_data) // UNKNOWN_CLASS_COUNT
    unknown_train_data, unknown_train_labels = [], []
    unknown_test_data, unknown_test_labels = [], []

    for class_idx in range(UNKNOWN_CLASS_COUNT):
        start_idx = class_idx * samples_per_unknown_class
        end_idx = start_idx + samples_per_unknown_class
        class_data = unknown_data[start_idx:end_idx]
        indices = torch.randperm(len(class_data))
        train_idx = indices[:4000]
        test_idx = indices[4000:5000]

        unknown_train_data.append(class_data[train_idx])
        unknown_test_data.append(class_data[test_idx])

        class_label = KNOWN_CLASS_COUNT + class_idx
        unknown_train_labels.append(torch.full((len(train_idx),), class_label, dtype=torch.long))
        unknown_test_labels.append(torch.full((len(test_idx),), class_label, dtype=torch.long))

    dataset = {
        'train_known_data': train_known_data,
        'train_known_labels': train_known_labels,
        'test_known_data': test_known_data,
        'test_known_labels': test_known_labels,
        'train_unknown_data': torch.cat(unknown_train_data, dim=0),
        'train_unknown_labels': torch.cat(unknown_train_labels, dim=0),
        'test_unknown_data': torch.cat(unknown_test_data, dim=0),
        'test_unknown_labels': torch.cat(unknown_test_labels, dim=0),
    }
    print(f"  未知类训练样本: {len(dataset['train_unknown_data'])} (每类4000)")
    print(f"  未知类测试样本: {len(dataset['test_unknown_data'])} (每类1000)")
    return dataset


def build_pretraining_loader(signals: torch.Tensor, batch_size: int = 256) -> DataLoader:
    dataset = TensorDataset(signals, signals, signals)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_feature_extractor(signals: torch.Tensor, device: torch.device, epochs: int = 20) -> GCDPretrainer:
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
    dataloader = build_pretraining_loader(signals)
    for epoch in range(epochs):
        loss = pretrainer.train_epoch(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Reconstruction Loss: {loss:.6f}")
    return pretrainer


def reconstruct_samples(pretrainer: GCDPretrainer, samples: torch.Tensor, device: torch.device) -> torch.Tensor:
    pretrainer.backbone.eval()
    pretrainer.reconstruction_head.eval()
    with torch.no_grad():
        signals = samples.to(device)
        features = pretrainer.backbone(signals)
        reconstructed, _ = pretrainer.reconstruction_head(features)
    return reconstructed.cpu()


def plot_reconstructions_per_class(
    class_id: int,
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    samples_per_figure: int = 10,
) -> None:
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
    fig.savefig(RECON_DIR / f"class_{class_id + 1:02d}_samples.png", dpi=300)
    plt.close(fig)


def visualize_reconstruction_per_class(
    train_signals: torch.Tensor,
    train_labels: torch.Tensor,
    pretrainer: GCDPretrainer,
    device: torch.device,
    samples_per_class: int = 10,
) -> None:
    print("\nReconstructing first 10 samples per class...")
    for class_id in range(KNOWN_CLASS_COUNT):
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
            samples_per_figure=samples_per_class,
        )
    print(f"Visualization saved to: {RECON_DIR}")


def build_test_set(dataset):
    test_data = torch.cat([dataset['test_known_data'], dataset['test_unknown_data']], dim=0)
    test_labels = torch.cat([dataset['test_known_labels'], dataset['test_unknown_labels']], dim=0)
    if test_data.dim() == 2:
        test_data = test_data.unsqueeze(1)
    return test_data, test_labels


def evaluate_open_set(pretrainer: GCDPretrainer, device: torch.device, dataset):
    test_data, test_labels = build_test_set(dataset)

    print("\n=== 5. 特征提取 ===")
    pretrainer.backbone.eval()
    with torch.no_grad():
        test_features = pretrainer.backbone(test_data.to(device)).cpu().numpy()
    print(f"特征提取完成: {test_features.shape}")

    print("\n=== 6. UMAP降维 ===")
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings = umap_reducer.fit_transform(test_features)
    print(f"UMAP降维完成: {embeddings.shape}")

    print("\n=== 7. DBSCAN聚类 ===")
    dbscan = DBSCAN(eps=0.15, min_samples=7)
    cluster_labels = dbscan.fit_predict(test_features)
    unique_clusters = np.unique(cluster_labels)
    noise_count = np.sum(cluster_labels == -1)
    print(f"发现聚类数量: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
    print(f"噪声点数量: {noise_count} ({noise_count/len(cluster_labels)*100:.1f}%)")

    print("\n=== 8. 开集性能分析 ===")
    test_labels_np = test_labels.numpy()
    known_mask = test_labels_np < KNOWN_CLASS_COUNT
    unknown_mask = ~known_mask

    if np.any(known_mask):
        known_ari = adjusted_rand_score(test_labels_np[known_mask], cluster_labels[known_mask])
        known_nmi = normalized_mutual_info_score(test_labels_np[known_mask], cluster_labels[known_mask])
        print(f"已知类 ARI: {known_ari:.4f}")
        print(f"已知类 NMI: {known_nmi:.4f}")
    else:
        known_ari = known_nmi = float('nan')

    if np.any(unknown_mask):
        unknown_unique_clusters = len(np.unique(cluster_labels[unknown_mask]))
        unknown_noise_ratio = np.sum(cluster_labels[unknown_mask] == -1) / np.sum(unknown_mask)
        print(f"未知类聚类数量: {unknown_unique_clusters}")
        print(f"未知类噪声比例: {unknown_noise_ratio:.4f}")
    else:
        unknown_unique_clusters = 0
        unknown_noise_ratio = float('nan')

    confusion_matrix, pred_headers = build_confusion_matrix(test_labels_np, cluster_labels)
    overall_accuracy, known_accuracy, unknown_accuracy, noise_ratio = compute_confusion_metrics(confusion_matrix)

    print("\n聚类混淆矩阵 (真实标签 vs 预测簇/噪声):")
    header_row = "      " + "  ".join(f"{h:>6}" for h in pred_headers)
    print(header_row)
    for row_idx in range(TOTAL_CLASSES):
        row_label = f"T{row_idx+1:>2}"
        row_values = "  ".join(f"{confusion_matrix[row_idx, col_idx]:>6}" for col_idx in range(len(pred_headers)))
        print(f"{row_label}  {row_values}")

    confusion_matrix_markdown = confusion_matrix_to_markdown(confusion_matrix, pred_headers)

    plot_umap_open_set(embeddings, test_labels_np, cluster_labels, save_dir=OPEN_SET_DIR)
    save_open_set_outputs(test_features, embeddings, test_labels_np, cluster_labels, known_ari, known_nmi, unknown_unique_clusters, unknown_noise_ratio, overall_accuracy, known_accuracy, unknown_accuracy, noise_ratio, confusion_matrix_markdown)


def build_confusion_matrix(true_labels: np.ndarray, cluster_labels: np.ndarray):
    pred_headers = [f"K{i+1}" for i in range(KNOWN_CLASS_COUNT)] + [f"U{i+KNOWN_CLASS_COUNT+1}" for i in range(UNKNOWN_CLASS_COUNT)] + ["Noise"]
    confusion_matrix = np.zeros((TOTAL_CLASSES, len(pred_headers)), dtype=int)

    cluster_to_pred = {}
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            cluster_to_pred[cluster_id] = len(pred_headers) - 1
            continue
        mask = cluster_labels == cluster_id
        if not np.any(mask):
            cluster_to_pred[cluster_id] = len(pred_headers) - 1
            continue
        labels_in_cluster = true_labels[mask]
        counts = np.bincount(labels_in_cluster, minlength=TOTAL_CLASSES)
        majority_label = int(np.argmax(counts))
        cluster_to_pred[cluster_id] = majority_label

    for true_label, cluster_label in zip(true_labels, cluster_labels):
        pred_index = cluster_to_pred.get(cluster_label, len(pred_headers) - 1)
        confusion_matrix[true_label, pred_index] += 1

    return confusion_matrix, pred_headers


def compute_confusion_metrics(confusion_matrix: np.ndarray):
    total_samples = confusion_matrix.sum()
    diag_sum = sum(confusion_matrix[i, i] for i in range(KNOWN_CLASS_COUNT + UNKNOWN_CLASS_COUNT))
    overall_accuracy = diag_sum / total_samples if total_samples > 0 else 0.0

    known_total = confusion_matrix[:KNOWN_CLASS_COUNT, :].sum()
    known_correct = sum(confusion_matrix[i, i] for i in range(KNOWN_CLASS_COUNT))
    known_accuracy = known_correct / known_total if known_total > 0 else 0.0

    unknown_total = confusion_matrix[KNOWN_CLASS_COUNT:, :].sum()
    unknown_correct = sum(confusion_matrix[KNOWN_CLASS_COUNT + i, KNOWN_CLASS_COUNT + i] for i in range(UNKNOWN_CLASS_COUNT))
    unknown_accuracy = unknown_correct / unknown_total if unknown_total > 0 else 0.0

    noise_total = confusion_matrix[:, -1].sum()
    noise_ratio = noise_total / total_samples if total_samples > 0 else 0.0

    return overall_accuracy, known_accuracy, unknown_accuracy, noise_ratio


def confusion_matrix_to_markdown(confusion_matrix: np.ndarray, headers):
    lines = []
    header = "| 真实\\预测 | " + " | ".join(headers) + " |"
    sep = "|" + " --- |" * (len(headers) + 1)
    lines.append(header)
    lines.append(sep)
    for row_idx in range(TOTAL_CLASSES):
        row_name = f"类{row_idx+1}"
        row_cells = " | ".join(str(confusion_matrix[row_idx, col_idx]) for col_idx in range(len(headers)))
        lines.append(f"| {row_name} | {row_cells} |")
    return "\n".join(lines)


def save_open_set_outputs(features, embeddings, true_labels, cluster_labels, known_ari, known_nmi, unknown_clusters, unknown_noise_ratio, overall_accuracy, known_accuracy, unknown_accuracy, noise_ratio, confusion_matrix_markdown):
    np.save(OPEN_SET_DIR / "true_open_set_features.npy", features)
    np.save(OPEN_SET_DIR / "true_open_set_embeddings.npy", embeddings)
    np.save(OPEN_SET_DIR / "true_open_set_true_labels.npy", true_labels)
    np.save(OPEN_SET_DIR / "true_open_set_cluster_labels.npy", cluster_labels)

    report = f"""
# data1 开集分类分析报告

## 数据配置
- 已知类别: {KNOWN_CLASS_COUNT} (1-{KNOWN_CLASS_COUNT})
- 未知类别: {UNKNOWN_CLASS_COUNT} ({KNOWN_CLASS_COUNT+1}-{TOTAL_CLASSES})

## 预训练
- 模型: CA1D Backbone + Reconstruction Head
- 特征维度: 128
- 学习率: 1e-4

## 聚类结果
- 已知类 ARI: {known_ari:.4f}
- 已知类 NMI: {known_nmi:.4f}
- 未知类聚类数量: {unknown_clusters}
- 未知类噪声比例: {unknown_noise_ratio:.4f}

## 指标
- 整体准确率: {overall_accuracy:.4f}
- 已知类准确率: {known_accuracy:.4f}
- 未知类识别准确率: {unknown_accuracy:.4f}
- 噪声占比: {noise_ratio:.4f}

## 混淆矩阵
{confusion_matrix_markdown}
"""
    with open(OPEN_SET_DIR / "true_open_set_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"结果保存到: {OPEN_SET_DIR}")


def main():
    print("=" * 80)
    print("data1 数据集：特征提取 + 重构 + 开集分类")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    dataset = create_open_set_dataset()

    train_signals = dataset['train_known_data']
    train_labels = dataset['train_known_labels']
    if train_signals.dim() == 2:
        train_signals = train_signals.unsqueeze(1)

    print("\n=== 3. 自监督预训练 ===")
    pretrainer = train_feature_extractor(train_signals, device)

    visualize_reconstruction_per_class(train_signals, train_labels, pretrainer, device, samples_per_class=10)

    evaluate_open_set(pretrainer, device, dataset)


if __name__ == "__main__":
    main()
