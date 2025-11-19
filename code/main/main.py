#!/usr/bin/env python3
"""Open-set feature extraction, reconstruction, and visualization for data1."""

import sys
import warnings
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gcd_implementation.gcd_solver import GCDPretrainer
from src.solver.dataset import load_radar_known_fixed_test, load_radar_unknown

DATA_ROOT = REPO_ROOT / "data1" / "data_noise_30"
KNOWN_CLASS_COUNT = 7
UNKNOWN_CLASS_COUNT = 3
TOTAL_CLASSES = KNOWN_CLASS_COUNT + UNKNOWN_CLASS_COUNT

RECON_BATCH_SIZE = 256
SUPERVISED_CONTRASTIVE_WEIGHT = 1.0
SUPERVISED_CONTRASTIVE_TEMPERATURE = 0.2
SUPERVISED_CE_WEIGHT = 1.0
SUPERVISED_EPOCHS = 15
SUPERVISED_BATCH_SIZE = 128
SUPERVISED_LR = 1e-4
FREEZE_BACKBONE_IN_SUP = False

RECON_DIR = REPO_ROOT / "result" / "MIX" / "reconstruction_outputs_30"
RECON_DIR.mkdir(parents=True, exist_ok=True)
OPEN_SET_DIR = REPO_ROOT / "result" / "MIX" / "true_open_set_results_data1_30"
OPEN_SET_DIR.mkdir(parents=True, exist_ok=True)


def plot_umap_open_set(embeddings, true_labels, cluster_labels, save_dir=None):
    # 固定风格并绘制四个子图，便于在同一嵌入上对比不同可视化结果。
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 为已知/未知类别预先分配固定颜色，让图例保持一致。
    colors_known = plt.cm.tab10(np.linspace(0, 1, KNOWN_CLASS_COUNT))
    colors_unknown = plt.cm.Set3(np.linspace(0, 1, UNKNOWN_CLASS_COUNT))

    # 子图 (0,0)：逐个绘制已知类别，观察 UMAP 后的分离度。
    for i in range(KNOWN_CLASS_COUNT):
        mask = true_labels == i
        axes[0, 0].scatter(embeddings[mask, 0], embeddings[mask, 1],
                           c=[colors_known[i]], s=20, alpha=0.7, label=f'Known Class {i+1}')

    # 未知类别使用不同标记叠加，增强对比度。
    for i in range(UNKNOWN_CLASS_COUNT):
        mask = true_labels == (KNOWN_CLASS_COUNT + i)
        axes[0, 0].scatter(embeddings[mask, 0], embeddings[mask, 1],
                           c=[colors_unknown[i]], s=30, alpha=0.7, marker='^', label=f'Unknown Class {KNOWN_CLASS_COUNT + i + 1}')

    axes[0, 0].set_title('True Labels (Known vs Unknown)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('UMAP 1')
    axes[0, 0].set_ylabel('UMAP 2')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # 为聚类结果准备调色板（含噪声标签 -1）。
    unique_clusters = np.unique(cluster_labels)
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    # 子图 (0,1)：展示 KMeans 聚类；若算法输出噪声标签亦会单独标记。
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

    # 子图 (1,0)：压缩为已知/未知二分类视图，观察整体趋势。
    known_mask = true_labels < KNOWN_CLASS_COUNT
    unknown_mask = ~known_mask
    axes[1, 0].scatter(embeddings[known_mask, 0], embeddings[known_mask, 1], c='blue', s=15, alpha=0.5, label='Known Classes')
    axes[1, 0].scatter(embeddings[unknown_mask, 0], embeddings[unknown_mask, 1], c='red', s=25, alpha=0.7, marker='^', label='Unknown Classes')
    axes[1, 0].set_title('Known vs Unknown Classes', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    axes[1, 0].legend()

    # 子图 (1,1)：统计各个聚类的样本数量，查看混合情况。
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
    # 以确定性顺序加载已知类别的数据集，保证评估可复现。
    train_loader_known, test_loader_known = load_radar_known_fixed_test(
        root=str(DATA_ROOT),
        known_class_count=KNOWN_CLASS_COUNT,
        test_per_class=1000,
        batch_size=256,
        shuffle_train=False,
        shuffle_test=False,
        as_loader=True,
    )

    # 未知样本单独加载，便于构造若干伪未知类别。
    unknown_loader = load_radar_unknown(
        root=str(DATA_ROOT),
        known_class_count=KNOWN_CLASS_COUNT,
        batch_size=256,
        as_loader=True,
    )

    # 将数据加载器展开成张量，以支持随机切片而不是按批次顺序。
    train_batches = [(data, labels) for data, labels in train_loader_known]
    test_batches = [(data, labels) for data, labels in test_loader_known]
    unknown_batches = [(data, labels) for data, labels in unknown_loader]

    # 拼接所有批次，方便做索引和统计。
    train_known_data = torch.cat([data for data, _ in train_batches], dim=0)
    train_known_labels = torch.cat([labels for _, labels in train_batches], dim=0)

    test_known_data = torch.cat([data for data, _ in test_batches], dim=0)
    test_known_labels = torch.cat([labels for _, labels in test_batches], dim=0)

    unknown_data = torch.cat([data for data, _ in unknown_batches], dim=0)

    print(f"  已知类训练样本: {len(train_known_data)}")
    print(f"  已知类测试样本: {len(test_known_data)}")
    print(f"  未知类样本总数: {len(unknown_data)}")

    # 计算每个伪未知类别应分配的样本数量。
    samples_per_unknown_class = len(unknown_data) // UNKNOWN_CLASS_COUNT
    unknown_train_data, unknown_train_labels = [], []
    unknown_test_data, unknown_test_labels = [], []

    for class_idx in range(UNKNOWN_CLASS_COUNT):
        # 对每个伪类别随机拆分训练与测试子集。
        start_idx = class_idx * samples_per_unknown_class
        end_idx = start_idx + samples_per_unknown_class
        class_data = unknown_data[start_idx:end_idx]
        indices = torch.randperm(len(class_data))
        train_idx = indices[:4000]
        test_idx = indices[4000:5000]

        # 将切分后的张量暂存，循环结束后再统一拼接。
        unknown_train_data.append(class_data[train_idx])
        unknown_test_data.append(class_data[test_idx])

        class_label = KNOWN_CLASS_COUNT + class_idx
        unknown_train_labels.append(torch.full((len(train_idx),), class_label, dtype=torch.long))
        unknown_test_labels.append(torch.full((len(test_idx),), class_label, dtype=torch.long))

    # 将结果统一打包成字典，简化调用方逻辑。
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


def ensure_channel_dim(signals: torch.Tensor) -> torch.Tensor:
    """确保输入含有 [B, C, L] 形状，如果缺少通道维则补齐。"""
    if signals.dim() == 2:
        return signals.unsqueeze(1)
    return signals


def build_supervised_loader(signals: torch.Tensor, labels: torch.Tensor, batch_size: int = 256, shuffle: bool = True) -> DataLoader:
    """构建用于监督分类的 DataLoader。"""
    dataset = TensorDataset(signals, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate_classifier(backbone: nn.Module, classifier: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """在指定数据集上评估分类准确率。"""
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
    pretrainer: GCDPretrainer,
    train_signals: torch.Tensor,
    train_labels: torch.Tensor,
    val_signals: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    num_classes: int,
    *,
    batch_size: int = 128,
    epochs: int = 15,
    lr: float = 1e-4,
    freeze_backbone: bool = False,
) -> Tuple[nn.Module, dict]:
    """使用交叉熵在已知类别上训练一个分类头。"""
    backbone = pretrainer.backbone
    train_loader = build_supervised_loader(train_signals, train_labels, batch_size=batch_size, shuffle=True)
    val_loader = build_supervised_loader(val_signals, val_labels, batch_size=batch_size, shuffle=False)

    # 推理一次确定 backbone 的输出维度。
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
    """使用 batch 内类均值作为原型的交叉熵，不引入显式分类头。"""
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


def train_feature_extractor(
    signals: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    contrastive_weight: float = SUPERVISED_CONTRASTIVE_WEIGHT,
    ce_weight: float = SUPERVISED_CE_WEIGHT,
) -> GCDPretrainer:
    # 根据信号长度和通道数配置 backbone，确保尺寸匹配。
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
    dataloader = DataLoader(dataset, batch_size=RECON_BATCH_SIZE, shuffle=True)

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
                features, batch_labels, SUPERVISED_CONTRASTIVE_TEMPERATURE
            )

            ce_loss = prototype_cross_entropy(features, batch_labels)

            loss = recon_loss + contrastive_weight * sup_contrast_loss + ce_weight * ce_loss
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
    # 将网络切换到 eval 模式，保证 BN/Dropout 行为稳定。
    pretrainer.backbone.eval()
    pretrainer.reconstruction_head.eval()
    with torch.no_grad():
        # 将样本搬到设备，编码得到特征再解码出重构结果。
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
    # 确保原始与重构张量的维度完全一致。
    assert original.shape == reconstructed.shape
    # 转成 NumPy 供 Matplotlib 使用，并压缩掉通道维。
    original_np = original.squeeze(1).cpu().numpy()
    reconstructed_np = reconstructed.squeeze(1).cpu().numpy()
    time_axis = np.arange(original_np.shape[1])

    rows, cols = 5, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows), sharex=True)
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx < min(samples_per_figure, original_np.shape[0]):
            # 叠加原始与重构波形，直观比较差异。
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

    # 每个类别输出一张拼图，方便快速检查。
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
    # 遍历所有已知类别，抽样查看若干重构。
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
    # 合并已知与未知测试集，使评估覆盖完整开放集。
    test_data = torch.cat([dataset['test_known_data'], dataset['test_unknown_data']], dim=0)
    test_labels = torch.cat([dataset['test_known_labels'], dataset['test_unknown_labels']], dim=0)
    if test_data.dim() == 2:
        # 若缺少通道维则补齐，以满足 1D 卷积输入。
        test_data = test_data.unsqueeze(1)
    return test_data, test_labels


def evaluate_open_set(pretrainer: GCDPretrainer, device: torch.device, dataset):
    # 在合并后的测试集上执行完整开放集识别流程。
    test_data, test_labels = build_test_set(dataset)

    print("\n=== 5. 特征提取 ===")
    pretrainer.backbone.eval()
    with torch.no_grad():
        # 仅提取一次特征，后续步骤复用缓存结果。
        test_features = pretrainer.backbone(test_data.to(device)).cpu().numpy()
    print(f"特征提取完成: {test_features.shape}")

    print("\n=== 6. UMAP降维 ===")
    # 使用 UMAP 降维以保持局部结构并便于可视化。
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings = umap_reducer.fit_transform(test_features)
    print(f"UMAP降维完成: {embeddings.shape}")

    print("\n=== 7. KMeans聚类 ===")
    # 使用 KMeans 聚类，目标团簇数为已知+未知全部类别。
    kmeans = KMeans(n_clusters=TOTAL_CLASSES, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(test_features)
    unique_clusters = np.unique(cluster_labels)
    print(f"KMeans 聚类数量: {len(unique_clusters)}")

    print("\n=== 8. 开集性能分析 ===")
    # 按已知/未知拆分评估指标并分别输出。
    test_labels_np = test_labels.numpy()
    known_mask = test_labels_np < KNOWN_CLASS_COUNT
    unknown_mask = ~known_mask

    if np.any(known_mask):
        # ARI/NMI 用于衡量聚类对已知标签的恢复程度。
        known_ari = adjusted_rand_score(test_labels_np[known_mask], cluster_labels[known_mask])
        known_nmi = normalized_mutual_info_score(test_labels_np[known_mask], cluster_labels[known_mask])
        print(f"已知类 ARI: {known_ari:.4f}")
        print(f"已知类 NMI: {known_nmi:.4f}")
    else:
        known_ari = known_nmi = float('nan')

    if np.any(unknown_mask):
        # 统计未知样本对应的聚类数量与噪声占比，评估分离情况。
        unknown_unique_clusters = len(np.unique(cluster_labels[unknown_mask]))
        unknown_noise_ratio = np.sum(cluster_labels[unknown_mask] == -1) / np.sum(unknown_mask)
        print(f"未知类聚类数量: {unknown_unique_clusters}")
        print(f"未知类噪声比例: {unknown_noise_ratio:.4f}")
    else:
        unknown_unique_clusters = 0
        unknown_noise_ratio = float('nan')

    # 通过聚类多数投票映射回类别，构建混淆矩阵。
    confusion_matrix, pred_headers = build_confusion_matrix(test_labels_np, cluster_labels)
    overall_accuracy, known_accuracy, unknown_accuracy, noise_ratio = compute_confusion_metrics(confusion_matrix)

    print("\n聚类混淆矩阵 (真实标签 vs 预测簇/噪声):")
    header_row = "      " + "  ".join(f"{h:>6}" for h in pred_headers)
    print(header_row)
    for row_idx in range(TOTAL_CLASSES):
        row_label = f"T{row_idx+1:>2}"
        row_values = "  ".join(f"{confusion_matrix[row_idx, col_idx]:>6}" for col_idx in range(len(pred_headers)))
        print(f"{row_label}  {row_values}")

    # 将混淆矩阵同时打印并写入 Markdown 报告。
    confusion_matrix_markdown = confusion_matrix_to_markdown(confusion_matrix, pred_headers)

    # 保存可视化与序列化数组，便于线下分析。
    plot_umap_open_set(embeddings, test_labels_np, cluster_labels, save_dir=OPEN_SET_DIR)
    save_open_set_outputs(test_features, embeddings, test_labels_np, cluster_labels, known_ari, known_nmi, unknown_unique_clusters, unknown_noise_ratio, overall_accuracy, known_accuracy, unknown_accuracy, noise_ratio, confusion_matrix_markdown)


def build_confusion_matrix(true_labels: np.ndarray, cluster_labels: np.ndarray):
    # 列对应所有已知类别、未知伪类别以及一个噪声列。
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
        # 使用聚类内的多数标签决定该聚类映射到的预测类别。
        labels_in_cluster = true_labels[mask]
        counts = np.bincount(labels_in_cluster, minlength=TOTAL_CLASSES)
        majority_label = int(np.argmax(counts))
        cluster_to_pred[cluster_id] = majority_label

    for true_label, cluster_label in zip(true_labels, cluster_labels):
        # 根据样本真实标签和预测聚类，为对应单元格加一。
        pred_index = cluster_to_pred.get(cluster_label, len(pred_headers) - 1)
        confusion_matrix[true_label, pred_index] += 1

    return confusion_matrix, pred_headers


def compute_confusion_metrics(confusion_matrix: np.ndarray):
    total_samples = confusion_matrix.sum()
    diag_sum = sum(confusion_matrix[i, i] for i in range(KNOWN_CLASS_COUNT + UNKNOWN_CLASS_COUNT))
    overall_accuracy = diag_sum / total_samples if total_samples > 0 else 0.0

    # 已知类别准确率只统计前 KNOWN_CLASS_COUNT 行。
    known_total = confusion_matrix[:KNOWN_CLASS_COUNT, :].sum()
    known_correct = sum(confusion_matrix[i, i] for i in range(KNOWN_CLASS_COUNT))
    known_accuracy = known_correct / known_total if known_total > 0 else 0.0

    # 未知准确率只统计伪未知类别所在的行。
    unknown_total = confusion_matrix[KNOWN_CLASS_COUNT:, :].sum()
    unknown_correct = sum(confusion_matrix[KNOWN_CLASS_COUNT + i, KNOWN_CLASS_COUNT + i] for i in range(UNKNOWN_CLASS_COUNT))
    unknown_accuracy = unknown_correct / unknown_total if unknown_total > 0 else 0.0

    # 噪声占比用来衡量聚类算法未归属的样本比例（KMeans 通常为 0）。
    noise_total = confusion_matrix[:, -1].sum()
    noise_ratio = noise_total / total_samples if total_samples > 0 else 0.0

    return overall_accuracy, known_accuracy, unknown_accuracy, noise_ratio


def confusion_matrix_to_markdown(confusion_matrix: np.ndarray, headers):
    # 以 Markdown 表格渲染混淆矩阵，方便写入报告。
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
    # 保存特征等原始产物，避免重复运行耗时流程。
    np.save(OPEN_SET_DIR / "true_open_set_features.npy", features)
    np.save(OPEN_SET_DIR / "true_open_set_embeddings.npy", embeddings)
    np.save(OPEN_SET_DIR / "true_open_set_true_labels.npy", true_labels)
    np.save(OPEN_SET_DIR / "true_open_set_cluster_labels.npy", cluster_labels)

    # 汇总所有指标生成 Markdown 报告。
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
    # 将报告写入磁盘以备查阅。
    with open(OPEN_SET_DIR / "true_open_set_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"结果保存到: {OPEN_SET_DIR}")


def main():
    print("=" * 80)
    print("data1 数据集：特征提取 + 重构 + 开集分类")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 统一使用同一张设备
    print(f"使用设备: {device}")

    # 构建已知/未知数据集，保留原始顺序
    dataset = create_open_set_dataset()  

    # 取出已知类别的信号和标签，用于重构式预训练。
    train_signals = dataset['train_known_data']
    train_labels = dataset['train_known_labels']
    if train_signals.dim() == 2:
        train_signals = train_signals.unsqueeze(1)  # CA1D backbone 需要 [B, C, L]

    print("\n=== 3. 自监督预训练 ===")
    # 训练 backbone + 重构头
    pretrainer = train_feature_extractor(train_signals, train_labels, device)  

    # 针对每个类别取前 10 条样本，展示原始 vs 重构对比
    # 在执行开放集评估前，先导出每类的重构对比图。
    visualize_reconstruction_per_class(
        train_signals,
        train_labels,
        pretrainer,
        device,
        samples_per_class=10,
    )

    # 使用预训练 backbone 提特征，并完成 UMAP+KMeans 的开集分析
    # 对合并后的测试集执行聚类与指标计算流程。
    evaluate_open_set(pretrainer, device, dataset)


if __name__ == "__main__":
    main()
