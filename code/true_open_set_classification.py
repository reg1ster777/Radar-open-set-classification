#!/usr/bin/env python3
"""
真正的开集分类实现
严格按照要求的数据划分：
- 训练集：已知类(1-7类)每类4000样本 + 未知类(8-10类)每类40样本(无标签)
- 测试集：已知类(1-7类)每类1000样本 + 未知类(8-10类)每类10样本(无标签)
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import umap
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.solver.dataset import load_radar_known_fixed_test, load_radar_unknown
from src.gcd_implementation.gcd_solver import GCDPretrainer

def plot_umap_open_set(embeddings, true_labels, cluster_labels, known_classes=7, total_classes=10, save_dir=None):
    """绘制开集分类的UMAP可视化"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 创建颜色映射
    colors_known = plt.cm.tab10(np.linspace(0, 1, known_classes))
    colors_unknown = plt.cm.Set3(np.linspace(0, 1, total_classes - known_classes))

    # 1. 真实标签（已知类和未知类用不同颜色和标记）
    for i in range(known_classes):
        mask = true_labels == i
        axes[0, 0].scatter(embeddings[mask, 0], embeddings[mask, 1],
                          c=[colors_known[i]], s=20, alpha=0.7, label=f'Known Class {i+1}')

    for i in range(known_classes, total_classes):
        mask = true_labels == i
        axes[0, 0].scatter(embeddings[mask, 0], embeddings[mask, 1],
                          c=[colors_unknown[i-known_classes]], s=30, alpha=0.7,
                          marker='^', label=f'Unknown Class {i+1}')

    axes[0, 0].set_title('True Labels (Known vs Unknown)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('UMAP 1')
    axes[0, 0].set_ylabel('UMAP 2')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # 2. 聚类结果
    unique_clusters = np.unique(cluster_labels)
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        if cluster_id == -1:  # 噪声点
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

    # 3. 已知类 vs 未知类对比
    known_mask = true_labels < known_classes
    unknown_mask = ~known_mask

    axes[1, 0].scatter(embeddings[known_mask, 0], embeddings[known_mask, 1],
                      c='blue', s=15, alpha=0.5, label='Known Classes')
    axes[1, 0].scatter(embeddings[unknown_mask, 0], embeddings[unknown_mask, 1],
                      c='red', s=25, alpha=0.7, marker='^', label='Unknown Classes')

    axes[1, 0].set_title('Known vs Unknown Classes', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    axes[1, 0].legend()

    # 4. 聚类统计
    cluster_sizes = []
    cluster_labels_unique = []
    for cluster_id in unique_clusters:
        if cluster_id != -1:
            size = np.sum(cluster_labels == cluster_id)
            cluster_sizes.append(size)
            cluster_labels_unique.append(f'C{cluster_id}')

    axes[1, 1].bar(cluster_labels_unique, cluster_sizes)
    axes[1, 1].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Cluster ID')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "true_open_set_umap_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_open_set_dataset():
    """创建符合要求的开集数据集"""
    print("创建开集数据集...")

    # 1. 加载已知类数据
    print("  加载已知类数据...")
    train_loader_known, test_loader_known = load_radar_known_fixed_test(
        root="data\LFM_dataset\data_noise_50",
        known_class_count=7,
        test_per_class=1000,
        batch_size=256,  # 使用合理的batch size
        as_loader=True
    )

    # 2. 加载未知类数据
    print("  加载未知类数据...")
    unknown_loader = load_radar_unknown(
        root="data\LFM_dataset\data_noise_50",
        known_class_count=7,
        batch_size=256,
        as_loader=True
    )

    # 3. 提取数据
    # 已知类训练数据 (28000样本)
    train_known_data_list = []
    train_known_labels_list = []
    for batch_data, batch_labels in train_loader_known:
        train_known_data_list.append(batch_data)
        train_known_labels_list.append(batch_labels)

    train_known_data = torch.cat(train_known_data_list, dim=0)
    train_known_labels = torch.cat(train_known_labels_list, dim=0)

    # 已知类测试数据 (7000样本)
    test_known_data_list = []
    test_known_labels_list = []
    for batch_data, batch_labels in test_loader_known:
        test_known_data_list.append(batch_data)
        test_known_labels_list.append(batch_labels)

    test_known_data = torch.cat(test_known_data_list, dim=0)
    test_known_labels = torch.cat(test_known_labels_list, dim=0)

    # 未知类数据 (15000样本)
    unknown_data_list = []
    unknown_labels_list = []
    for batch_data, batch_labels in unknown_loader:
        unknown_data_list.append(batch_data)
        unknown_labels_list.append(batch_labels)

    unknown_data = torch.cat(unknown_data_list, dim=0)
    unknown_labels = torch.cat(unknown_labels_list, dim=0)

    print(f"  已知类训练数据: {len(train_known_data)} 样本")
    print(f"  已知类测试数据: {len(test_known_data)} 样本")
    print(f"  未知类数据: {len(unknown_data)} 样本")

    # 4. 按照要求重新划分未知类数据
    # 未知类训练数据：每类4000样本 (共12000样本)
    # 未知类测试数据：每类1000样本 (共3000样本)

    # 假设未知类数据按顺序排列，每类5000样本
    unknown_train_data = []
    unknown_test_data = []

    for class_idx in range(3):  # 3个未知类
        start_idx = class_idx * 5000
        end_idx = start_idx + 5000
        class_data = unknown_data[start_idx:end_idx]

        # 随机选择
        indices = torch.randperm(len(class_data))
        train_indices = indices[:4000]   # 4000个用于训练
        test_indices = indices[4000:5000]  # 1000个用于测试

        unknown_train_data.append(class_data[train_indices])
        unknown_test_data.append(class_data[test_indices])

    unknown_train_data = torch.cat(unknown_train_data, dim=0)
    unknown_test_data = torch.cat(unknown_test_data, dim=0)

    print(f"  未知类训练数据: {len(unknown_train_data)} 样本 (每类4000)")
    print(f"  未知类测试数据: {len(unknown_test_data)} 样本 (每类1000)")

    return {
        'train_known_data': train_known_data,           # 7*4000
        'train_known_labels': train_known_labels,       # 7*4000
        'train_unknown_data': unknown_train_data,       # 3*4000
        'test_known_data': test_known_data,             # 7*1000
        'test_known_labels': test_known_labels,         # 7*1000
        'test_unknown_data': unknown_test_data          # 3*1000
    }

def main():
    print("=" * 80)
    print("真正的开集分类实现")
    print("数据划分：已知类(1-7) + 未知类(8-10)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    known_class_count = 7
    unknown_class_count = 3
    total_classes = known_class_count + unknown_class_count
    all_epoch = 1

    # === 1. 创建数据集 ===
    print("\n" + "=" * 50)
    print("=== 1. 数据集创建 ===")
    print("=" * 50)

    dataset = create_open_set_dataset()

    print('dataset["train_known_data"].shape:', dataset["train_known_data"].shape)
    print('dataset["train_known_labels"].shape:', dataset["train_known_labels"].shape)
    print('dataset["train_unknown_data"].shape:', dataset["train_unknown_data"].shape)
    print('dataset["test_known_data"].shape:', dataset["test_known_data"].shape)
    print('dataset["test_known_labels"].shape:', dataset["test_known_labels"].shape)
    print('dataset["test_unknown_data"].shape:', dataset["test_unknown_data"].shape)

    # === 2. 构建训练数据 ===
    print("\n" + "=" * 50)
    print("=== 2. 构建训练数据 ===")
    print("=" * 50)

    # # 合并训练数据：已知类(有标签) + 未知类(无标签)
    # train_data = torch.cat([
    #     dataset['train_known_data'],
    #     dataset['train_unknown_data']
    # ], dim=0)
    # # 已知类保持原标签，未知类标签设为-1
    # train_labels = torch.cat([
    #     dataset['train_known_labels'],
    #     torch.full((len(dataset['train_unknown_data']),), -1, dtype=torch.long)
    # ], dim=0)

    # 仅使用已知类数据作为训练集
    train_data = dataset['train_known_data']
    train_labels = dataset['train_known_labels']

    # 转换数据格式为 [batch, channels, length]
    print("train_data.shape:",train_data.shape)

    if train_data.dim() == 2:
        train_data = train_data.unsqueeze(1)

    print("train_data.shape:",train_data.shape)

    print(f"总训练数据: {len(train_data)} 样本")
    print(f"已知类样本: {torch.sum(train_labels >= 0).item()}")
    print(f"未知类样本: {torch.sum(train_labels == -1).item()}")

    # === 3. 自监督预训练 ===
    print("\n" + "=" * 50)
    print("=== 3. 自监督预训练 ===")
    print("=" * 50)

    # 创建预训练器
    pretrainer = GCDPretrainer(
        arch_type="ca1d",
        original_signal_length=200,
        input_channels=1,
        feature_dim=128,
        device=device,
        lr=1e-4,
        compression_ratio=0.25
    )

    # 准备训练数据 (格式: signals, weak_aug, strong_aug)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_data, train_data)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True
    )

    # 预训练
    print("开始自监督预训练...")
    for epoch in range(all_epoch):
        loss = pretrainer.train_epoch(train_dataloader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{all_epoch}, Loss: {loss:.6f}")

    # 保存模型
    output_dir = Path("true_open_set_results")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "true_open_set_pretrained_backbone.pth"
    pretrainer.save_backbone(model_path)
    print(f"预训练模型已保存: {model_path}")

    # === 4. 构建测试数据 ===
    print("\n" + "=" * 50)
    print("=== 4. 构建测试数据 ===")
    print("=" * 50)

    # 测试集包含所有10个类别
    # 从完整的未知类数据中提取每类1000条样本用于测试
    # 需要重新加载完整的未知类数据
    print("  重新加载完整的未知类数据用于测试集构建...")

    # 加载完整的未知类数据
    unknown_loader_full = load_radar_unknown(
        root="data\LFM_dataset\data_noise_50",
        known_class_count=7,
        batch_size=256,
        as_loader=True
    )

    # 提取所有未知类数据
    unknown_full_data_list = []
    unknown_full_labels_list = []
    for batch_data, batch_labels in unknown_loader_full:
        unknown_full_data_list.append(batch_data)
        unknown_full_labels_list.append(batch_labels)

    unknown_full_data = torch.cat(unknown_full_data_list, dim=0)
    unknown_full_labels = torch.cat(unknown_full_labels_list, dim=0)

    print(f"  完整未知类数据: {len(unknown_full_data)} 样本")

    # 为每个未知类提取1000条测试样本
    # 未知类数据没有标签，按照数据顺序划分（假设每类5000条数据）
    unknown_test_data_list = []
    unknown_test_labels_list = []

    samples_per_unknown_class = len(unknown_full_data) // 3  # 15000 // 3 = 5000

    for class_idx in range(3):  # 3个未知类
        # 根据数据顺序获取当前类的数据
        start_idx = class_idx * samples_per_unknown_class
        end_idx = start_idx + samples_per_unknown_class
        class_data = unknown_full_data[start_idx:end_idx]

        # 随机选择1000条样本
        if len(class_data) >= 1000:
            indices = torch.randperm(len(class_data))[:1000]
            selected_data = class_data[indices]
        else:
            # 如果数据不足1000条，使用全部数据
            selected_data = class_data
            print(f"    警告: 未知类 {class_idx+8} 只有 {len(class_data)} 条数据，不足1000条")

        unknown_test_data_list.append(selected_data)
        unknown_test_labels_list.extend([7 + class_idx] * len(selected_data))

    unknown_test_data = torch.cat(unknown_test_data_list, dim=0)
    unknown_test_labels = torch.tensor(unknown_test_labels_list, dtype=torch.long)

    # 合并已知类和未知类测试数据
    test_data = torch.cat([
        dataset['test_known_data'],
        unknown_test_data
    ], dim=0)

    test_labels = torch.cat([
        dataset['test_known_labels'],
        unknown_test_labels
    ], dim=0)

    # 转换数据格式
    if test_data.dim() == 2:
        test_data = test_data.unsqueeze(1)

    print(f"总测试数据: {len(test_data)} 样本")
    print(f"已知类测试样本: {len(dataset['test_known_data'])} (7类，每类1000)")
    print(f"未知类测试样本: {len(unknown_test_data)} (3类，每类1000条)")
    print(f"测试集包含所有10个类别，总计10*1000条数据")

    # 训练用的未知类数据
    print(f"未知类训练样本: {len(dataset['train_unknown_data'])} (用于预训练)")

    # === 5. 特征提取 ===
    print("\n" + "=" * 50)
    print("=== 5. 特征提取 ===")
    print("=" * 50)

    pretrainer.backbone.eval()
    with torch.no_grad():
        test_features = pretrainer.backbone(test_data.to(device)).cpu().numpy()

    print(f"特征提取完成: {test_features.shape[0]} 个样本, {test_features.shape[1]} 维特征")

    # === 6. UMAP降维 ===
    print("\n" + "=" * 50)
    print("=== 6. UMAP降维 ===")
    print("=" * 50)

    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings = umap_reducer.fit_transform(test_features)

    print(f"UMAP降维完成: {embeddings.shape}")

    # === 7. DBSCAN聚类 ===
    print("\n" + "=" * 50)
    print("=== 7. DBSCAN聚类 ===")
    print("=" * 50)

    dbscan = DBSCAN(eps=0.15, min_samples=7)
    cluster_labels = dbscan.fit_predict(test_features)

    unique_clusters = np.unique(cluster_labels)
    noise_count = np.sum(cluster_labels == -1)

    print(f"发现聚类数量: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
    print(f"噪声点数量: {noise_count} ({noise_count/len(cluster_labels)*100:.1f}%)")

    # === 8. 开集性能分析 ===
    print("\n" + "=" * 50)
    print("=== 8. 开集性能分析 ===")
    print("=" * 50)

    # 分析已知类和未知类的聚类表现
    test_labels_np = test_labels.numpy()
    known_mask = test_labels_np < 7
    unknown_mask = ~known_mask

    known_ari = float("nan")
    known_nmi = float("nan")

    # 已知类聚类质量
    if np.sum(known_mask) > 0:
        known_ari = adjusted_rand_score(test_labels_np[known_mask], cluster_labels[known_mask])
        known_nmi = normalized_mutual_info_score(test_labels_np[known_mask], cluster_labels[known_mask])
        print(f"已知类聚类质量:")
        print(f"  ARI: {known_ari:.4f}")
        print(f"  NMI: {known_nmi:.4f}")

    # 未知类发现情况
    unknown_unique_clusters = 0
    unknown_noise_ratio = 0.0
    if np.sum(unknown_mask) > 0:
        unknown_unique_clusters = len(np.unique(cluster_labels[unknown_mask]))
        unknown_noise_ratio = np.sum(cluster_labels[unknown_mask] == -1) / np.sum(unknown_mask)
        print(f"未知类发现情况:")
        print(f"  发现聚类数: {unknown_unique_clusters}")
        print(f"  噪声比例: {unknown_noise_ratio:.4f}")

    # 整体聚类质量
    unique_true_labels = len(np.unique(test_labels_np))
    unique_cluster_labels = len(np.unique(cluster_labels[cluster_labels != -1]))

    print(f"\n整体聚类分析:")
    print(f"  真实类别数: {unique_true_labels}")
    print(f"  发现聚类数: {unique_cluster_labels}")
    print(f"  噪声点数: {np.sum(cluster_labels == -1)}")
    print(f"  总样本数: {len(test_labels_np)}")

    # 每个类别的聚类质量
    print(f"\n各类别聚类质量:")
    for class_id in range(unique_true_labels):
        mask = test_labels_np == class_id
        if np.sum(mask) > 0:
            class_cluster_labels = cluster_labels[mask]
            # 计算该类别的主要聚类
            unique_clusters, counts = np.unique(class_cluster_labels, return_counts=True)
            main_cluster = unique_clusters[np.argmax(counts)]
            purity = np.max(counts) / np.sum(counts)
            class_type = "已知类" if class_id < 7 else "未知类"
            print(f"  {class_type} {class_id+1}: 主要聚类={main_cluster}, 纯度={purity:.4f}, 样本数={np.sum(mask)}")

    # === 8.1 聚类混淆矩阵（真实标签 × 预测簇） ===
    num_true_classes = total_classes
    pred_headers = (
        [f"K{i+1}" for i in range(known_class_count)] +
        [f"U{i+known_class_count+1}" for i in range(unknown_class_count)] +
        ["Noise"]
    )
    confusion_matrix = np.zeros((num_true_classes, len(pred_headers)), dtype=int)

    # 将每个聚类标签映射到预测类别索引（噪声为最后一列）
    cluster_to_pred_index = {}
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            cluster_to_pred_index[cluster_id] = len(pred_headers) - 1
            continue

        cluster_mask = cluster_labels == cluster_id
        if not np.any(cluster_mask):
            cluster_to_pred_index[cluster_id] = len(pred_headers) - 1
            continue

        labels_in_cluster = test_labels_np[cluster_mask]
        label_counts = np.bincount(labels_in_cluster, minlength=num_true_classes)
        majority_label = int(np.argmax(label_counts))
        cluster_to_pred_index[cluster_id] = majority_label

    # 累计混淆矩阵
    for true_label, cluster_label in zip(test_labels_np, cluster_labels):
        pred_index = cluster_to_pred_index.get(cluster_label, len(pred_headers) - 1)
        confusion_matrix[true_label, pred_index] += 1

    print("\n聚类混淆矩阵 (行=真实标签, 列=预测标签/噪声):")
    header_row = "      " + "  ".join(f"{h:>6}" for h in pred_headers)
    print(header_row)
    for row_idx in range(num_true_classes):
        row_label = f"T{row_idx+1:>2}"
        row_values = "  ".join(f"{confusion_matrix[row_idx, col_idx]:>6}" for col_idx in range(len(pred_headers)))
        print(f"{row_label}  {row_values}")

    # 计算基础指标
    total_samples = confusion_matrix.sum()
    diag_sum = sum(confusion_matrix[i, i] for i in range(min(num_true_classes, len(pred_headers) - 1)))
    overall_accuracy = diag_sum / total_samples if total_samples > 0 else 0.0

    known_total = confusion_matrix[:known_class_count, :].sum()
    known_correct = sum(confusion_matrix[i, i] for i in range(known_class_count))
    known_accuracy = known_correct / known_total if known_total > 0 else 0.0

    unknown_total = confusion_matrix[known_class_count:, :].sum()
    unknown_correct = sum(
        confusion_matrix[known_class_count + i, known_class_count + i]
        for i in range(unknown_class_count)
    )
    unknown_accuracy = unknown_correct / unknown_total if unknown_total > 0 else 0.0

    noise_total = confusion_matrix[:, -1].sum()
    noise_ratio = noise_total / total_samples if total_samples > 0 else 0.0

    print(f"\n整体准确率: {overall_accuracy:.4f}")
    print(f"已知类准确率: {known_accuracy:.4f}")
    print(f"未知类识别准确率: {unknown_accuracy:.4f}")
    print(f"噪声占比: {noise_ratio:.4f} ({noise_total}/{total_samples})")

    # 构建 Markdown 表格
    cm_markdown_lines = []
    markdown_header = "| 真实\\预测 | " + " | ".join(pred_headers) + " |"
    markdown_sep = "|" + " --- |" * (len(pred_headers) + 1)
    cm_markdown_lines.append(markdown_header)
    cm_markdown_lines.append(markdown_sep)
    for row_idx in range(num_true_classes):
        row_name = f"类 {row_idx+1}"
        row_cells = " | ".join(str(confusion_matrix[row_idx, col_idx]) for col_idx in range(len(pred_headers)))
        cm_markdown_lines.append(f"| {row_name} | {row_cells} |")
    confusion_matrix_markdown = "\n".join(cm_markdown_lines)

    # === 9. 可视化 ===
    print("\n" + "=" * 50)
    print("=== 9. 可视化 ===")
    print("=" * 50)

    plot_umap_open_set(
        embeddings,
        test_labels.numpy(),
        cluster_labels,
        known_classes=7,
        total_classes=10,
        save_dir=output_dir
    )

    # === 10. 保存结果 ===
    print("\n" + "=" * 50)
    print("=== 10. 保存结果 ===")
    print("=" * 50)

    # 保存数据
    np.save(output_dir / "true_open_set_features.npy", test_features)
    np.save(output_dir / "true_open_set_embeddings.npy", embeddings)
    np.save(output_dir / "true_open_set_true_labels.npy", test_labels.numpy())
    np.save(output_dir / "true_open_set_cluster_labels.npy", cluster_labels)

    # 生成报告
    report = f"""
# 真正的开集分类分析报告

## 数据配置
- 已知类别数: 7 (类1-7)
- 未知类别数: 3 (类8-10)

## 训练数据
- 已知类样本: {len(dataset['train_known_data'])} (每类4000)
- 未知类样本: {len(dataset['train_unknown_data'])} (每类4000，无标签，未参与训练)
- 总训练样本: {len(train_data)}

## 测试数据
- 已知类样本: {len(dataset['test_known_data'])} (每类1000)
- 未知类样本: {len(unknown_test_data)} (每类1000，无标签)
- 总测试样本: {len(test_data)} (10*1000条数据)

## 预训练配置
- 骨干网络: ca1d
- 特征维度: 128
- 预训练轮数: 50
- 学习率: 1e-4

## 聚类结果
- 发现聚类数: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}
- 噪声点数: {noise_count} ({noise_count/len(cluster_labels)*100:.1f}%)

## 开集识别性能
- 已知类ARI: {known_ari:.4f}
- 已知类NMI: {known_nmi:.4f}
- 未知类发现聚类数: {unknown_unique_clusters}
- 未知类噪声比例: {unknown_noise_ratio:.4f}

## 评估指标
- 整体准确率: {overall_accuracy:.4f}
- 已知类准确率: {known_accuracy:.4f}
- 未知类识别准确率: {unknown_accuracy:.4f}
- 噪声占比: {noise_ratio:.4f} ({noise_total}/{total_samples})

## 聚类混淆矩阵
{confusion_matrix_markdown}

## 文件输出
- 特征文件: true_open_set_features.npy
- UMAP嵌入: true_open_set_embeddings.npy
- 真实标签: true_open_set_true_labels.npy
- 聚类标签: true_open_set_cluster_labels.npy
- 可视化图: true_open_set_umap_analysis.png
- 预训练模型: true_open_set_pretrained_backbone.pth
"""

    with open(output_dir / "true_open_set_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 50)
    print("开集分类分析完成!")
    print("=" * 50)
    print(f"结果保存在: {output_dir}/")
    print("可视化图: true_open_set_umap_analysis.png")
    print("分析报告: true_open_set_analysis_report.md")

if __name__ == "__main__":
    main()
