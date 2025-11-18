#!/usr/bin/env python3
"""
主聚类分析脚本
在阶段一预训练的基础上进行K-means聚类和UMAP可视化
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

from src.clustering_analysis.feature_extractor import FeatureExtractor, create_dataloader_for_extraction
from src.clustering_analysis.clustering import ClusteringAnalysis
from src.clustering_analysis.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(description="聚类分析和可视化")
    parser.add_argument("--data_root", type=str, default="data/LFM_dataset/data_noise_50",
                       help="数据根目录")
    parser.add_argument("--model_path", type=str, default="",
                       help="预训练模型路径（可选）")
    parser.add_argument("--arch_type", type=str, default="ca1d", choices=["ca1d", "hybrid", "mamba"],
                       help="骨干网络架构")
    parser.add_argument("--known_class_count", type=int, default=7,
                       help="已知类别数量")
    parser.add_argument("--total_classes", type=int, default=10,
                       help="总类别数量")
    parser.add_argument("--n_clusters", type=int, default=10,
                       help="聚类数量")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="批次大小")
    parser.add_argument("--random_state", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--output_dir", type=str, default="clustering_results",
                       help="输出目录")

    # UMAP参数
    parser.add_argument("--umap_neighbors", type=int, default=15,
                       help="UMAP邻居数量")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                       help="UMAP最小距离参数")

    # 分析选项
    parser.add_argument("--find_optimal_k", action="store_true",
                       help="是否搜索最优聚类数")
    parser.add_argument("--compare_methods", action="store_true",
                       help="是否对比不同降维方法")
    parser.add_argument("--save_all_plots", action="store_true",
                       help="是否保存所有图表")

    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录: {output_dir}")

    # === 1. 数据加载和特征提取 ===
    print("\n" + "="*50)
    print("=== 1. 数据加载和特征提取 ===")
    print("="*50)

    # 创建数据加载器
    dataloader, dataset = create_dataloader_for_extraction(
        args.data_root, args.known_class_count, args.batch_size
    )

    # 加载预训练模型
    if args.model_path and Path(args.model_path).exists():
        backbone = FeatureExtractor.load_pretrained_backbone(
            args.model_path, args.arch_type, device=device
        )
    else:
        print("⚠️  使用未初始化的模型（随机权重）")
        from src.gcd_implementation.models import get_backbone
        backbone, actual_feature_dim = get_backbone(args.arch_type)
        backbone = backbone.to(device)
        print(f"骨干网络特征维度: {actual_feature_dim}")

    # 提取特征
    extractor = FeatureExtractor(backbone, device)
    features, labels, original_labels = extractor.extract_features(dataloader)

    # 保存特征
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "original_labels.npy", original_labels)
    print("特征数据已保存")

    # === 2. K-means聚类 ===
    print("\n" + "="*50)
    print("=== 2. K-means聚类 ===")
    print("="*50)

    # 创建聚类分析器
    cluster_analyzer = ClusteringAnalysis(
        n_clusters=args.n_clusters,
        random_state=args.random_state
    )

    # 执行聚类
    clustering_results = cluster_analyzer.fit_kmeans(features)

    # 评估聚类质量
    clustering_metrics = cluster_analyzer.evaluate_clustering(features, labels, original_labels)

    # 分析聚类分布
    distribution_df = cluster_analyzer.analyze_cluster_distribution(labels, original_labels)
    print("\n聚类分布分析:")
    print(distribution_df.head(15))

    # 保存聚类结果
    np.save(output_dir / "cluster_labels.npy", cluster_analyzer.cluster_labels)
    distribution_df.to_csv(output_dir / "cluster_distribution.csv", index=False)
    print("聚类结果已保存")

    # === 3. UMAP可视化 ===
    print("\n" + "="*50)
    print("=== 3. UMAP可视化 ===")
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

    # 绘制真实标签分布
    if args.save_all_plots:
        visualizer.plot_embedding_with_labels(
            labels,
            title="True Labels Distribution",
            save_path=output_dir / "umap_true_labels.png"
        )

    # 绘制聚类结果对比
    visualizer.plot_embedding_with_clusters(
        cluster_analyzer.cluster_labels,
        labels,
        title="K-means Clustering Results",
        save_path=output_dir / "umap_clustering_comparison.png" if args.save_all_plots else None
    )

    # === 4. 高级分析 ===
    print("\n" + "="*50)
    print("=== 4. 高级分析 ===")
    print("="*50)

    # 搜索最优聚类数
    if args.find_optimal_k:
        print("\n搜索最优聚类数...")
        optimal_k_results = cluster_analyzer.find_optimal_k(
            features,
            k_range=(2, min(15, len(features)//10)),
            plot=True,
            save_path=output_dir / "optimal_k_analysis.png" if args.save_all_plots else None
        )

        # 找到轮廓系数最高的K
        best_k_idx = np.argmax(optimal_k_results['silhouette_scores'])
        best_k = optimal_k_results['k_values'][best_k_idx]
        best_silhouette = optimal_k_results['silhouette_scores'][best_k_idx]
        print(f"推荐最优聚类数: K={best_k} (轮廓系数: {best_silhouette:.4f})")

    # 对比不同降维方法
    if args.compare_methods:
        print("\n对比不同降维方法...")
        visualizer.plot_embedding_comparisons(
            features, labels,
            save_dir=output_dir if args.save_all_plots else None
        )

    # 分析降维质量
    embedding_quality = visualizer.analyze_embedding_quality(features, labels)

    # === 5. 生成报告 ===
    print("\n" + "="*50)
    print("=== 5. 分析报告 ===")
    print("="*50)

    report = f"""
# 聚类分析报告

## 数据信息
- 数据源: {args.data_root}
- 预训练模型: {args.model_path}
- 骨干网络: {args.arch_type}
- 样本数量: {len(features)}
- 特征维度: {features.shape[1]}
- 真实类别数: {len(np.unique(labels))}
- 聚类数量: {args.n_clusters}

## 聚类质量指标
- ARI (调整兰德指数): {clustering_metrics['adjusted_rand_score']:.4f}
- NMI (标准化互信息): {clustering_metrics['normalized_mutual_info_score']:.4f}
- 轮廓系数: {clustering_metrics.get('silhouette_score', 'N/A'):.4f}
- 平均聚类纯度: {clustering_metrics['avg_purity']:.4f}
- 惯性值 (Inertia): {clustering_results['inertia']:.2f}

## 降维质量 ({visualizer.embedding_method})
- 平均邻居重合率: {embedding_quality['avg_neighbor_overlap']:.4f}

## 聚类分布
- 聚类大小范围: {min(clustering_metrics['cluster_sizes'])} - {max(clustering_metrics['cluster_sizes'])}
- 真实类别大小: {clustering_metrics['true_class_sizes']}

## 参数设置
- UMAP邻居数: {args.umap_neighbors}
- UMAP最小距离: {args.umap_min_dist}
- 随机种子: {args.random_state}
"""

    print(report)

    # 保存报告
    with open(output_dir / "analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ 分析完成! 所有结果已保存到: {output_dir}")

    # 创建结果摘要
    summary = {
        'n_samples': len(features),
        'feature_dim': features.shape[1],
        'n_true_classes': len(np.unique(labels)),
        'n_clusters': args.n_clusters,
        'ari': clustering_metrics['adjusted_rand_score'],
        'nmi': clustering_metrics['normalized_mutual_info_score'],
        'silhouette_score': clustering_metrics.get('silhouette_score', 0.0),
        'avg_purity': clustering_metrics['avg_purity'],
        'inertia': clustering_results['inertia'],
        'umap_neighbor_overlap': embedding_quality['avg_neighbor_overlap']
    }

    import json
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("📊 摘要统计:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()