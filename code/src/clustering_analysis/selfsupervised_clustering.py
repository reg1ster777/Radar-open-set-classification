#!/usr/bin/env python3
"""
正确的自监督聚类分析流程
1. 划分训练集/测试集
2. 在训练集上自监督预训练
3. 用预训练模型提取测试集特征
4. DBSCAN聚类和UMAP可视化
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gcd_implementation.datasets import GCDDataset
from src.gcd_implementation.gcd_solver import GCDPretrainer
from src.gcd_implementation.models import get_backbone
from src.clustering_analysis.clustering import ClusteringAnalysis
from src.clustering_analysis.visualization import Visualizer
from torch.utils.data import DataLoader, Subset, random_split


def create_train_test_split(dataset: GCDDataset, train_ratio: float = 0.8,
                           random_state: int = 42) -> tuple:
    """
    创建训练集和测试集分割

    Args:
        dataset: 完整数据集
        train_ratio: 训练集比例
        random_state: 随机种子

    Returns:
        train_dataset, test_dataset
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    # 确保可重复的结果
    generator = torch.Generator()
    generator.manual_seed(random_state)

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    print(f"数据集划分: 训练集 {train_size:,} 样本, 测试集 {test_size:,} 样本")

    return train_dataset, test_dataset


def analyze_dataset_distribution(dataset, dataset_name="数据集"):
    """分析数据集分布"""
    labels = []
    original_labels = []

    # 统计前1000个样本的分布（避免内存问题）
    sample_size = min(1000, len(dataset))
    for i in range(sample_size):
        _, label, original_label = dataset[i]
        labels.append(label.item())
        original_labels.append(int(original_label))

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    unique_orig_labels, orig_label_counts = np.unique(original_labels, return_counts=True)

    print(f"\n{dataset_name}分布 (前{sample_size}样本):")
    print(f"  重映射标签: {dict(zip(unique_labels, label_counts))}")
    print(f"  原始标签: {dict(zip(unique_orig_labels, orig_label_counts))}")

    return unique_labels, unique_orig_labels


def selfsupervised_pretrain(train_dataset, arch_type: str = "ca1d",
                          device: torch.device = None, epochs: int = 50,
                          batch_size: int = 256, lr: float = 1e-4,
                          model_save_path: str = None) -> torch.nn.Module:
    """
    自监督预训练

    Args:
        train_dataset: 训练数据集
        arch_type: 架构类型
        device: 计算设备
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        model_save_path: 模型保存路径

    Returns:
        预训练的骨干网络
    """
    print(f"\n=== 自监督预训练 ===")
    print(f"架构: {arch_type}, 轮数: {epochs}, 批次大小: {batch_size}, 学习率: {lr}")

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)

    # 创建预训练器
    pretrainer = GCDPretrainer(
        arch_type=arch_type,
        original_signal_length=200,  # 实际信号长度
        input_channels=1,
        feature_dim=128,
        device=device,
        lr=lr,
        compression_ratio=0.25
    )

    # 训练循环
    print("开始自监督预训练...")
    for epoch in range(epochs):
        avg_loss = pretrainer.train_epoch(train_dataloader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # 保存模型
    if model_save_path:
        model_save_path = Path(model_save_path)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        pretrainer.save_backbone(model_save_path)
        print(f"预训练模型已保存: {model_save_path}")

    return pretrainer.backbone


def extract_features_with_pretrained_model(backbone: torch.nn.Module,
                                         test_dataset,
                                         device: torch.device = None,
                                         batch_size: int = 256) -> tuple:
    """
    用预训练模型提取测试集特征

    Args:
        backbone: 预训练的骨干网络
        test_dataset: 测试数据集
        device: 计算设备
        batch_size: 批次大小

    Returns:
        features, labels, original_labels
    """
    print(f"\n=== 特征提取 ===")

    from src.clustering_analysis.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor(backbone, device)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=0)

    features, labels, original_labels = extractor.extract_features(test_dataloader)

    print(f"特征提取完成: {features.shape[0]} 个样本, {features.shape[1]} 维特征")

    return features, labels, original_labels


def main():
    parser = argparse.ArgumentParser(description="自监督聚类分析")
    parser.add_argument("--data_root", type=str, default="data/LFM_dataset/data_noise_50",
                       help="数据根目录")
    parser.add_argument("--arch_type", type=str, default="ca1d", choices=["ca1d", "hybrid", "mamba"],
                       help="骨干网络架构")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="训练集比例")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="批次大小")
    parser.add_argument("--epochs", type=int, default=50,
                       help="预训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--eps", type=float, default=0.5,
                       help="DBSCAN的邻域半径参数")
    parser.add_argument("--min_samples", type=int, default=5,
                       help="DBSCAN的核心点最小邻居数")
    parser.add_argument("--random_state", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--output_dir", type=str, default="selfsupervised_clustering_results",
                       help="输出目录")
    parser.add_argument("--model_save_path", type=str, default="pretrained_models/selfsupervised_backbone.pth",
                       help="预训练模型保存路径")

    # UMAP参数
    parser.add_argument("--umap_neighbors", type=int, default=15,
                       help="UMAP邻居数量")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                       help="UMAP最小距离参数")

    # 分析选项
    parser.add_argument("--save_plots", action="store_true",
                       help="是否保存图表")

    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录: {output_dir}")

    # === 1. 数据加载和划分 ===
    print("\n" + "="*50)
    print("=== 1. 数据加载和划分 ===")
    print("="*50)

    # 创建完整数据集
    full_dataset = GCDDataset(
        root=args.data_root,
        known_class_labels=list(range(1, 11)),  # 使用所有10个类别
        unlabeled_class_labels=list(range(1, 11))
    )

    print(f"完整数据集: {len(full_dataset):,} 个样本")

    # 分析完整数据集分布
    all_labels, all_orig_labels = analyze_dataset_distribution(full_dataset, "完整数据集")

    # 划分训练集和测试集
    train_dataset, test_dataset = create_train_test_split(
        full_dataset, args.train_ratio, args.random_state
    )

    # 分析训练集和测试集分布
    analyze_dataset_distribution(train_dataset, "训练集")
    analyze_dataset_distribution(test_dataset, "测试集")

    # === 2. 自监督预训练 ===
    print("\n" + "="*50)
    print("=== 2. 自监督预训练 ===")
    print("="*50)

    # 在训练集上进行自监督预训练
    pretrained_backbone = selfsupervised_pretrain(
        train_dataset,
        arch_type=args.arch_type,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_save_path=args.model_save_path
    )

    # === 3. 特征提取 ===
    print("\n" + "="*50)
    print("=== 3. 测试集特征提取 ===")
    print("="*50)

    # 用预训练模型提取测试集特征
    features, labels, original_labels = extract_features_with_pretrained_model(
        pretrained_backbone, test_dataset, device, args.batch_size
    )

    # 统计测试集标签分布
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    unique_orig_labels, orig_label_counts = np.unique(original_labels, return_counts=True)

    print(f"\n测试集最终统计:")
    print(f"  重映射标签分布: {dict(zip(unique_labels, label_counts))}")
    print(f"  原始标签分布: {dict(zip(unique_orig_labels, orig_label_counts))}")

    # 保存特征数据
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "original_labels.npy", original_labels)

    # === 4. DBSCAN聚类 ===
    print("\n" + "="*50)
    print("=== 4. DBSCAN聚类 ===")
    print("="*50)

    # 创建DBSCAN聚类分析器
    cluster_analyzer = ClusteringAnalysis(eps=args.eps, min_samples=args.min_samples)

    # 执行聚类
    clustering_results = cluster_analyzer.fit_dbscan(features)

    # 评估聚类质量
    clustering_metrics = cluster_analyzer.evaluate_clustering(features, labels, original_labels)

    # 分析聚类分布
    distribution_df = cluster_analyzer.analyze_cluster_distribution(labels, original_labels)
    print("\n聚类分布分析:")
    print(distribution_df.head(15))

    # 保存聚类结果
    np.save(output_dir / "cluster_labels.npy", cluster_analyzer.cluster_labels)
    distribution_df.to_csv(output_dir / "cluster_distribution.csv", index=False)

    # === 5. UMAP可视化 ===
    print("\n" + "="*50)
    print("=== 5. UMAP可视化 ===")
    print("="*50)

    # 创建可视化器
    visualizer = Visualizer(random_state=args.random_state)

    # UMAP降维
    embedding = visualizer.fit_umap(
        features,
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist
    )

    # 保存UMAP嵌入
    np.save(output_dir / "umap_embedding.npy", embedding)

    # 创建标签名称
    label_names = [f'Class {i}' for i in range(len(unique_labels))]

    # 绘制真实标签分布
    true_labels_plot_path = output_dir / "umap_true_labels.png" if args.save_plots else None
    visualizer.plot_embedding_with_labels(
        labels,
        title="Test Set - True Labels Distribution",
        label_names=label_names,
        save_path=true_labels_plot_path
    )

    # 绘制聚类结果对比
    clustering_plot_path = output_dir / "umap_clustering_comparison.png" if args.save_plots else None
    visualizer.plot_embedding_with_clusters(
        cluster_analyzer.cluster_labels, labels,
        title="Clustering Results Comparison",
        save_path=clustering_plot_path
    )

    # 分析降维质量
    embedding_quality = visualizer.analyze_embedding_quality(features, labels)

    # === 6. 生成报告 ===
    print("\n" + "="*50)
    print("=== 6. 分析报告 ===")
    print("="*50)

    report = f"""
# 自监督聚类分析报告

## 数据信息
- 数据源: {args.data_root}
- 训练集大小: {len(train_dataset):,}
- 测试集大小: {len(test_dataset):,}
- 训练比例: {args.train_ratio}
- 骨干网络: {args.arch_type}
- 预训练轮数: {args.epochs}

## 预训练信息
- 学习率: {args.lr}
- 批次大小: {args.batch_size}
- 模型保存路径: {args.model_save_path}

## 测试集信息
- 样本数量: {len(features)}
- 特征维度: {features.shape[1]}
- 真实类别数: {len(unique_orig_labels)}
- 各类别样本数: {dict(zip(unique_orig_labels, orig_label_counts))}

## 聚类质量指标
- ARI (调整兰德指数): {clustering_metrics['adjusted_rand_score']:.4f}
- NMI (标准化互信息): {clustering_metrics['normalized_mutual_info_score']:.4f}
- 轮廓系数: {clustering_metrics.get('silhouette_score', 'N/A'):.4f}
- 平均聚类纯度: {clustering_metrics['avg_purity']:.4f}
- 发现聚类数量: {clustering_results['n_clusters']}
- 噪声点数量: {clustering_results['noise_count']} ({clustering_results['noise_ratio']*100:.1f}%)
- DBSCAN参数: eps={args.eps}, min_samples={args.min_samples}

## 降维质量 ({visualizer.embedding_method})
- 平均邻居重合率: {embedding_quality['avg_neighbor_overlap']:.4f}

## UMAP参数
- 邻居数量: {args.umap_neighbors}
- 最小距离: {args.umap_min_dist}

## 随机种子
- 随机种子: {args.random_state}
"""

    print(report)

    # 保存报告
    with open(output_dir / "analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    # 创建结果摘要（确保所有数据类型可JSON序列化）
    def convert_numpy_types(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    summary = {
        'dataset_info': {
            'total_samples': len(full_dataset),
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'train_ratio': args.train_ratio
        },
        'pretraining': {
            'arch_type': args.arch_type,
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size
        },
        'clustering': {
            'n_samples': int(len(features)),
            'feature_dim': int(features.shape[1]),
            'n_true_classes': int(len(unique_orig_labels)),
            'n_clusters': int(clustering_results['n_clusters']),
            'noise_count': int(clustering_results['noise_count']),
            'noise_ratio': float(clustering_results['noise_ratio']),
            'ari': float(clustering_metrics['adjusted_rand_score']),
            'nmi': float(clustering_metrics['normalized_mutual_info_score']),
            'silhouette_score': float(clustering_metrics.get('silhouette_score', 0.0)),
            'avg_purity': float(clustering_metrics['avg_purity']),
            'dbscan_params': {
                'eps': float(args.eps),
                'min_samples': int(args.min_samples)
            }
        },
        'visualization': {
            'umap_neighbor_overlap': float(embedding_quality['avg_neighbor_overlap']),
            'umap_neighbors': int(args.umap_neighbors),
            'umap_min_dist': float(args.umap_min_dist)
        }
    }

    import json
    with open(output_dir / "summary.json", "w") as f:
        json.dump(convert_numpy_types(summary), f, indent=2)

    print(f"\n✅ 自监督聚类分析完成!")
    print(f"所有结果已保存到: {output_dir}")

    # 打印关键指标
    print(f"\n📊 关键指标:")
    print(f"  ARI: {summary['clustering']['ari']:.4f}")
    print(f"  NMI: {summary['clustering']['nmi']:.4f}")
    print(f"  轮廓系数: {summary['clustering']['silhouette_score']:.4f}")
    print(f"  平均纯度: {summary['clustering']['avg_purity']:.4f}")


if __name__ == "__main__":
    main()