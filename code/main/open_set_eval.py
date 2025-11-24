"""开集评估与结果持久化模块。

包含测试集构建、UMAP+KMeans 聚类评估、混淆矩阵生成以及输出保存。
"""

from __future__ import annotations

import numpy as np
import torch
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import config
from plotting import plot_umap_open_set


def build_test_set(dataset):
    """合并已知与未知测试样本，输出齐整的张量。"""
    test_data = torch.cat([dataset['test_known_data'], dataset['test_unknown_data']], dim=0)
    test_labels = torch.cat([dataset['test_known_labels'], dataset['test_unknown_labels']], dim=0)
    if test_data.dim() == 2:
        test_data = test_data.unsqueeze(1)
    return test_data, test_labels


def evaluate_open_set(pretrainer, device: torch.device, dataset):
    """执行 UMAP 降维 + KMeans 聚类，并打印/保存各项指标。"""
    test_data, test_labels = build_test_set(dataset)

    print("\n=== 5. 特征提取 ===")
    pretrainer.backbone.eval()
    with torch.no_grad():
        test_features = pretrainer.backbone(test_data.to(device)).cpu().numpy()
    print(f"特征提取结果: {test_features.shape}")

    print("\n=== 6. UMAP降维 ===")
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings = umap_reducer.fit_transform(test_features)
    print(f"UMAP降维结果: {embeddings.shape}")

    print("\n=== 7. KMeans聚类 ===")
    kmeans = KMeans(n_clusters=config.TOTAL_CLASSES, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(test_features)
    unique_clusters = np.unique(cluster_labels)
    print(f"KMeans 聚类数量: {len(unique_clusters)}")

    print("\n=== 8. 指标解析 ===")
    test_labels_np = test_labels.numpy()
    known_mask = test_labels_np < config.KNOWN_CLASS_COUNT
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
        unknown_noise_ratio = 0.0
        print(f"未知类聚类数: {unknown_unique_clusters}")
        print(f"未知噪声占比: {unknown_noise_ratio:.4f}")
    else:
        unknown_unique_clusters = 0
        unknown_noise_ratio = float('nan')

    confusion_matrix, pred_headers = build_confusion_matrix(test_labels_np, cluster_labels)
    overall_accuracy, known_accuracy, unknown_accuracy, noise_ratio = compute_confusion_metrics(confusion_matrix)

    print("\n混淆矩阵（真实标签 vs 聚类）:")
    header_row = "      " + "  ".join(f"{h:>6}" for h in pred_headers)
    print(header_row)
    for row_idx in range(config.TOTAL_CLASSES):
        row_label = f"T{row_idx+1:>2}"
        row_values = "  ".join(f"{confusion_matrix[row_idx, col_idx]:>6}" for col_idx in range(len(pred_headers)))
        print(f"{row_label}  {row_values}")

    confusion_matrix_markdown = confusion_matrix_to_markdown(confusion_matrix, pred_headers)

    plot_umap_open_set(embeddings, test_labels_np, cluster_labels, save_dir=config.OPEN_SET_DIR)
    save_open_set_outputs(
        test_features,
        embeddings,
        test_labels_np,
        cluster_labels,
        known_ari,
        known_nmi,
        unknown_unique_clusters,
        unknown_noise_ratio,
        overall_accuracy,
        known_accuracy,
        unknown_accuracy,
        noise_ratio,
        confusion_matrix_markdown,
    )


def build_confusion_matrix(true_labels: np.ndarray, cluster_labels: np.ndarray):
    """根据聚类结果构建兼容已知/未知+噪声的混淆矩阵。"""
    pred_headers = [f"K{i+1}" for i in range(config.KNOWN_CLASS_COUNT)] + [f"U{i+config.KNOWN_CLASS_COUNT+1}" for i in range(config.UNKNOWN_CLASS_COUNT)] + ["Noise"]
    confusion_matrix = np.zeros((config.TOTAL_CLASSES, len(pred_headers)), dtype=int)

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
        counts = np.bincount(labels_in_cluster, minlength=config.TOTAL_CLASSES)
        majority_label = int(np.argmax(counts))
        cluster_to_pred[cluster_id] = majority_label

    for true_label, cluster_label in zip(true_labels, cluster_labels):
        pred_index = cluster_to_pred.get(cluster_label, len(pred_headers) - 1)
        confusion_matrix[true_label, pred_index] += 1

    return confusion_matrix, pred_headers


def compute_confusion_metrics(confusion_matrix: np.ndarray):
    """计算总体/已知/未知准确率及噪声占比。"""
    total_samples = confusion_matrix.sum()
    diag_sum = sum(confusion_matrix[i, i] for i in range(config.KNOWN_CLASS_COUNT + config.UNKNOWN_CLASS_COUNT))
    overall_accuracy = diag_sum / total_samples if total_samples > 0 else 0.0

    known_total = confusion_matrix[:config.KNOWN_CLASS_COUNT, :].sum()
    known_correct = sum(confusion_matrix[i, i] for i in range(config.KNOWN_CLASS_COUNT))
    known_accuracy = known_correct / known_total if known_total > 0 else 0.0

    unknown_total = confusion_matrix[config.KNOWN_CLASS_COUNT:, :].sum()
    unknown_correct = sum(confusion_matrix[config.KNOWN_CLASS_COUNT + i, config.KNOWN_CLASS_COUNT + i] for i in range(config.UNKNOWN_CLASS_COUNT))
    unknown_accuracy = unknown_correct / unknown_total if unknown_total > 0 else 0.0

    noise_total = confusion_matrix[:, -1].sum()
    noise_ratio = noise_total / total_samples if total_samples > 0 else 0.0

    return overall_accuracy, known_accuracy, unknown_accuracy, noise_ratio


def confusion_matrix_to_markdown(confusion_matrix: np.ndarray, headers):
    """将混淆矩阵渲染为 Markdown 表格。"""
    lines = []
    header = "| 真实\\预测 | " + " | ".join(headers) + " |"
    sep = "|" + " --- |" * (len(headers) + 1)
    lines.append(header)
    lines.append(sep)
    for row_idx in range(config.TOTAL_CLASSES):
        row_name = f"类{row_idx+1}"
        row_cells = " | ".join(str(confusion_matrix[row_idx, col_idx]) for col_idx in range(len(headers)))
        lines.append(f"| {row_name} | {row_cells} |")
    return "\n".join(lines)


def save_open_set_outputs(
    features,
    embeddings,
    true_labels,
    cluster_labels,
    known_ari,
    known_nmi,
    unknown_clusters,
    unknown_noise_ratio,
    overall_accuracy,
    known_accuracy,
    unknown_accuracy,
    noise_ratio,
    confusion_matrix_markdown,
):
    """保存开集评估的各种结果，包括 numpy 数组与 Markdown 报告。"""
    np.save(config.OPEN_SET_DIR / "true_open_set_features.npy", features)
    np.save(config.OPEN_SET_DIR / "true_open_set_embeddings.npy", embeddings)
    np.save(config.OPEN_SET_DIR / "true_open_set_true_labels.npy", true_labels)
    np.save(config.OPEN_SET_DIR / "true_open_set_cluster_labels.npy", cluster_labels)

    report = f"""
# data1 开集分类分析

## 数据概况
- 已知类: {config.KNOWN_CLASS_COUNT} (1-{config.KNOWN_CLASS_COUNT})
- 未知类: {config.UNKNOWN_CLASS_COUNT} ({config.KNOWN_CLASS_COUNT+1}-{config.TOTAL_CLASSES})

## 评估
- 已知类 ARI: {known_ari:.4f}
- 已知类 NMI: {known_nmi:.4f}
- 未知类聚类数: {unknown_clusters}
- 未知噪声占比: {unknown_noise_ratio:.4f}

## 指标
- 总体准确率: {overall_accuracy:.4f}
- 已知类准确率: {known_accuracy:.4f}
- 未知识别准确率: {unknown_accuracy:.4f}
- 噪声占比: {noise_ratio:.4f}

## 混淆矩阵
{confusion_matrix_markdown}
"""
    with open(config.OPEN_SET_DIR / "true_open_set_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已保存至: {config.OPEN_SET_DIR}")
