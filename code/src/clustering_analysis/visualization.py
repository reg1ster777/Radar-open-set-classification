#!/usr/bin/env python3
"""
可视化模块
实现UMAP降维和多种可视化方式
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from typing import Optional, Tuple, List, Dict
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class Visualizer:
    """
    可视化器
    提供UMAP、t-SNE、PCA等降维方法和可视化功能
    """

    def __init__(self, random_state: int = 42):
        """
        初始化可视化器

        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        self.embedding = None
        self.embedding_method = None

    def fit_umap(self, features: np.ndarray, n_components: int = 2,
                 n_neighbors: int = 15, min_dist: float = 0.1,
                 metric: str = 'euclidean') -> np.ndarray:
        """
        使用UMAP进行降维

        Args:
            features: 特征矩阵 [N, feature_dim]
            n_components: 降维后的维度
            n_neighbors: UMAP邻居数量
            min_dist: 最小距离参数
            metric: 距离度量

        Returns:
            降维后的嵌入 [N, n_components]
        """
        print(f"UMAP降维: {features.shape} -> {n_components}D")
        print(f"  参数: n_neighbors={n_neighbors}, min_dist={min_dist}")

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=self.random_state
        )

        self.embedding = reducer.fit_transform(features)
        self.embedding_method = f'UMAP({n_neighbors},{min_dist})'

        print(f"✓ UMAP降维完成: {self.embedding.shape}")
        return self.embedding

    def fit_tsne(self, features: np.ndarray, n_components: int = 2,
                 perplexity: float = 30.0, learning_rate: float = 200.0) -> np.ndarray:
        """
        使用t-SNE进行降维

        Args:
            features: 特征矩阵
            n_components: 降维后的维度
            perplexity: 困惑度
            learning_rate: 学习率

        Returns:
            降维后的嵌入
        """
        print(f"t-SNE降维: {features.shape} -> {n_components}D")
        print(f"  参数: perplexity={perplexity}, learning_rate={learning_rate}")

        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=self.random_state,
            n_iter=1000
        )

        self.embedding = reducer.fit_transform(features)
        self.embedding_method = f'tSNE({perplexity})'

        print(f"✓ t-SNE降维完成: {self.embedding.shape}")
        return self.embedding

    def fit_pca(self, features: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        使用PCA进行降维

        Args:
            features: 特征矩阵
            n_components: 降维后的维度

        Returns:
            降维后的嵌入
        """
        print(f"PCA降维: {features.shape} -> {n_components}D")

        reducer = PCA(n_components=n_components, random_state=self.random_state)
        self.embedding = reducer.fit_transform(features)
        self.embedding_method = f'PCA({n_components})'

        explained_variance = reducer.explained_variance_ratio_
        print(f"✓ PCA降维完成: {self.embedding.shape}")
        print(f"  解释方差比: {explained_variance}")
        print(f"  累计解释方差: {explained_variance.sum():.4f}")

        return self.embedding

    def plot_embedding_with_labels(self, labels: np.ndarray,
                                   title: str = "Feature Visualization",
                                   label_names: Optional[List[str]] = None,
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (10, 8),
                                   alpha: float = 0.7):
        """
        绘制降维结果，用真实标签着色

        Args:
            labels: 标签数组
            title: 图表标题
            label_names: 标签名称列表
            save_path: 保存路径
            figsize: 图像大小
            alpha: 透明度
        """
        if self.embedding is None:
            raise ValueError("请先调用fit_umap/fit_tsne/fit_pca")

        plt.figure(figsize=figsize)

        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = label_names[i] if label_names else f'Class {label}'
            plt.scatter(self.embedding[mask, 0], self.embedding[mask, 1],
                       c=[colors[i]], label=label_name, alpha=alpha, s=30)

        plt.xlabel(f'{self.embedding_method} Dimension 1')
        plt.ylabel(f'{self.embedding_method} Dimension 2')
        plt.title(f'{title} ({self.embedding_method})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"标签可视化图已保存: {save_path}")

        plt.show()

    def plot_embedding_with_clusters(self, cluster_labels: np.ndarray,
                                     true_labels: np.ndarray,
                                     title: str = "Clustering Results",
                                     save_path: Optional[str] = None,
                                     figsize: Tuple[int, int] = (15, 6)):
        """
        并排显示聚类结果和真实标签

        Args:
            cluster_labels: 聚类标签
            true_labels: 真实标签
            title: 图表标题
            save_path: 保存路径
            figsize: 图像大小
        """
        if self.embedding is None:
            raise ValueError("请先调用fit_umap/fit_tsne/fit_pca")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 左图：聚类结果
        unique_clusters = np.unique(cluster_labels)
        colors_cluster = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            ax1.scatter(self.embedding[mask, 0], self.embedding[mask, 1],
                       c=[colors_cluster[i]], label=f'Cluster {cluster}', alpha=0.7, s=30)

        ax1.set_xlabel(f'{self.embedding_method} Dimension 1')
        ax1.set_ylabel(f'{self.embedding_method} Dimension 2')
        ax1.set_title(f'{title} - Clusters ({self.embedding_method})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图：真实标签
        unique_true = np.unique(true_labels)
        colors_true = plt.cm.tab10(np.linspace(0, 1, len(unique_true)))

        for i, label in enumerate(unique_true):
            mask = true_labels == label
            ax2.scatter(self.embedding[mask, 0], self.embedding[mask, 1],
                       c=[colors_true[i]], label=f'Class {label}', alpha=0.7, s=30)

        ax2.set_xlabel(f'{self.embedding_method} Dimension 1')
        ax2.set_ylabel(f'{self.embedding_method} Dimension 2')
        ax2.set_title(f'{title} - True Labels ({self.embedding_method})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"聚类对比图已保存: {save_path}")

        plt.show()

    def plot_embedding_comparisons(self, features: np.ndarray, labels: np.ndarray,
                                   save_dir: Optional[str] = None):
        """
        绘制多种降维方法的对比图

        Args:
            features: 特征矩阵
            labels: 标签数组
            save_dir: 保存目录
        """
        methods = [
            ('UMAP', self.fit_umap, {'n_neighbors': 15, 'min_dist': 0.1}),
            ('t-SNE', self.fit_tsne, {'perplexity': 30}),
            ('PCA', self.fit_pca, {})
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for idx, (method_name, fit_func, params) in enumerate(methods):
            # 执行降维
            embedding = fit_func(features, **params)

            # 绘制
            ax = axes[idx]
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                          c=[colors[i]], label=f'Class {label}', alpha=0.7, s=20)

            ax.set_xlabel(f'{method_name} Dimension 1')
            ax.set_ylabel(f'{method_name} Dimension 2')
            ax.set_title(f'{method_name} Visualization')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            save_path = f"{save_dir}/dimensionality_reduction_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"降维对比图已保存: {save_path}")

        plt.show()

    def analyze_embedding_quality(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        分析降维质量

        Args:
            features: 原始特征
            labels: 标签

        Returns:
            质量分析结果
        """
        if self.embedding is None:
            raise ValueError("请先调用降维方法")

        analysis = {}

        # 计算局部结构保持度
        from sklearn.neighbors import NearestNeighbors

        # 在原始特征空间找最近邻
        nn_original = NearestNeighbors(n_neighbors=11, metric='euclidean')
        nn_original.fit(features)
        distances_original, indices_original = nn_original.kneighbors(features)

        # 在降维空间找最近邻
        nn_embedded = NearestNeighbors(n_neighbors=11, metric='euclidean')
        nn_embedded.fit(self.embedding)
        distances_embedded, indices_embedded = nn_embedded.kneighbors(self.embedding)

        # 计算重合率（除了自己）
        overlap_rates = []
        for i in range(len(features)):
            original_neighbors = set(indices_original[i, 1:])  # 排除自己
            embedded_neighbors = set(indices_embedded[i, 1:])
            overlap = len(original_neighbors.intersection(embedded_neighbors))
            overlap_rate = overlap / 10.0  # 除以邻居数量
            overlap_rates.append(overlap_rate)

        analysis['avg_neighbor_overlap'] = np.mean(overlap_rates)
        analysis['method'] = self.embedding_method

        print(f"降维质量分析 ({self.embedding_method}):")
        print(f"  平均邻居重合率: {analysis['avg_neighbor_overlap']:.4f}")

        return analysis


if __name__ == "__main__":
    # 测试可视化器
    from sklearn.datasets import make_blobs

    # 创建模拟数据
    X, y = make_blobs(n_samples=1000, centers=10, n_features=128, random_state=42)

    # 创建可视化器
    visualizer = Visualizer()

    # 测试UMAP
    embedding = visualizer.fit_umap(X)
    visualizer.plot_embedding_with_labels(y, title="UMAP Test")

    # 测试多种方法对比
    visualizer.plot_embedding_comparisons(X, y)