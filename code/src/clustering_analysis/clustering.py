#!/usr/bin/env python3
"""
聚类分析模块
实现DBSCAN聚类和聚类质量评估
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import pandas as pd


class ClusteringAnalysis:
    """
    聚类分析器
    提供DBSCAN聚类和多种评估指标
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        初始化聚类分析器

        Args:
            eps: DBSCAN的邻域半径
            min_samples: 核心点的最小邻居数
        """
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = None
        self.cluster_labels = None
        self.n_clusters = None

    def fit_dbscan(self, features: np.ndarray) -> Dict:
        """
        执行DBSCAN聚类

        Args:
            features: 特征矩阵 [N, feature_dim]

        Returns:
            聚类结果字典
        """
        print(f"执行DBSCAN聚类: eps={self.eps}, min_samples={self.min_samples}, {features.shape[0]}个样本")

        # 创建DBSCAN模型
        self.dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='euclidean',
            n_jobs=-1
        )

        # 执行聚类
        self.cluster_labels = self.dbscan.fit_predict(features)

        # 计算聚类数量（不包括噪声点）
        unique_labels = np.unique(self.cluster_labels)
        self.n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_count = np.sum(self.cluster_labels == -1)

        # 计算聚类质量指标
        results = {
            'cluster_labels': self.cluster_labels,
            'n_clusters': self.n_clusters,
            'noise_count': noise_count,
            'noise_ratio': noise_count / len(features),
            'unique_labels': unique_labels.tolist()
        }

        print(f"聚类完成:")
        print(f"  发现聚类数量: {self.n_clusters}")
        print(f"  噪声点数量: {noise_count} ({noise_count/len(features)*100:.1f}%)")
        print(f"  有效标签范围: {unique_labels}")

        return results

    def evaluate_clustering(self, features: np.ndarray, true_labels: np.ndarray,
                           original_labels: Optional[np.ndarray] = None) -> Dict:
        """
        评估聚类质量

        Args:
            features: 特征矩阵
            true_labels: 真实标签（重映射后的）
            original_labels: 原始标签

        Returns:
            评估指标字典
        """
        if self.cluster_labels is None:
            raise ValueError("请先运行fit_dbscan()")

        print("计算聚类质量指标...")

        metrics = {}

        # 1. 轮廓系数 (不需要真实标签)
        if len(np.unique(self.cluster_labels)) > 1:  # 需要至少2个聚类
            silhouette_avg = silhouette_score(features, self.cluster_labels)
            metrics['silhouette_score'] = silhouette_avg
            print(f"  轮廓系数: {silhouette_avg:.4f}")

        # 2. ARI (需要真实标签)
        ari = adjusted_rand_score(true_labels, self.cluster_labels)
        metrics['adjusted_rand_score'] = ari
        print(f"  ARI (调整兰德指数): {ari:.4f}")

        # 3. NMI (需要真实标签)
        nmi = normalized_mutual_info_score(true_labels, self.cluster_labels)
        metrics['normalized_mutual_info_score'] = nmi
        print(f"  NMI (标准化互信息): {nmi:.4f}")

        # 4. 聚类分布统计
        cluster_counts = np.bincount(self.cluster_labels)
        metrics['cluster_sizes'] = cluster_counts.tolist()
        print(f"  聚类大小分布: {cluster_counts}")

        # 5. 真实类别分布
        true_counts = np.bincount(true_labels)
        metrics['true_class_sizes'] = true_counts.tolist()
        print(f"  真实类别大小分布: {true_counts}")

        # 6. 混淆矩阵
        confusion_mat = confusion_matrix(true_labels, self.cluster_labels)
        metrics['confusion_matrix'] = confusion_mat.tolist()

        # 7. 聚类纯度分析
        cluster_purities = []
        unique_clusters = np.unique(self.cluster_labels)

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # 跳过噪声点
                continue
            cluster_mask = self.cluster_labels == cluster_id
            if cluster_mask.sum() > 0:
                cluster_true_labels = true_labels[cluster_mask]
                most_common_count = np.bincount(cluster_true_labels).max()
                purity = most_common_count / cluster_mask.sum()
                cluster_purities.append(purity)

        if cluster_purities:  # 只在有有效聚类时计算平均值
            metrics['cluster_purities'] = cluster_purities
            metrics['avg_purity'] = np.mean(cluster_purities)
            print(f"  平均聚类纯度: {metrics['avg_purity']:.4f}")
        else:
            metrics['cluster_purities'] = []
            metrics['avg_purity'] = 0.0
            print(f"  未发现有效聚类，平均纯度: 0.0000")

        return metrics

    def analyze_cluster_distribution(self, true_labels: np.ndarray,
                                   original_labels: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        分析聚类中真实类别的分布

        Args:
            true_labels: 真实标签
            original_labels: 原始标签

        Returns:
            聚类分布DataFrame
        """
        if self.cluster_labels is None:
            raise ValueError("请先运行fit_dbscan()")

        cluster_analysis = []
        unique_clusters = np.unique(self.cluster_labels)

        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_true_labels = true_labels[cluster_mask]

            # 统计该聚类中各类别的数量
            unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
            total_samples = len(cluster_true_labels)

            for label, count in zip(unique_labels, counts):
                cluster_analysis.append({
                    'cluster_id': cluster_id,
                    'true_label': label,
                    'original_label': original_labels[cluster_mask][np.where(cluster_true_labels == label)[0]][0] if original_labels is not None else label,
                    'count': count,
                    'percentage': count / total_samples * 100,
                    'cluster_size': total_samples
                })

        df = pd.DataFrame(cluster_analysis)
        return df.sort_values(['cluster_id', 'count'], ascending=[True, False])

    def plot_cluster_distribution(self, true_labels: np.ndarray,
                                 save_path: Optional[str] = None):
        """
        绘制聚类分布热力图

        Args:
            true_labels: 真实标签
            save_path: 保存路径
        """
        if self.cluster_labels is None:
            raise ValueError("请先运行fit_dbscan()")

        # 创建混淆矩阵
        confusion_mat = confusion_matrix(true_labels, self.cluster_labels)

        # 绘制热力图
        plt.figure(figsize=(12, 8))

        # 动态生成聚类标签（包括噪声点）
        unique_clusters = sorted(np.unique(self.cluster_labels))
        cluster_labels = [f'Noise {i}' if i == -1 else f'Cluster {i}' for i in unique_clusters]

        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                   xticklabels=cluster_labels,
                   yticklabels=[f'Class {i}' for i in range(len(np.unique(true_labels)))])
        plt.title('DBSCAN Cluster vs True Class Distribution')
        plt.xlabel('Predicted Cluster')
        plt.ylabel('True Class')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"聚类分布图已保存: {save_path}")

        plt.show()

    def find_optimal_eps(self, features: np.ndarray, eps_range: Tuple[float, float] = (0.1, 2.0),
                       eps_steps: int = 20, min_samples_range: Tuple[int, int] = (3, 15),
                       plot: bool = True, save_path: Optional[str] = None) -> Dict:
        """
        寻找DBSCAN的最优参数组合

        Args:
            features: 特征矩阵
            eps_range: eps参数搜索范围
            eps_steps: eps测试步数
            min_samples_range: min_samples参数搜索范围
            plot: 是否绘制图表
            save_path: 保存路径

        Returns:
            不同参数组合的结果
        """
        eps_values = np.linspace(eps_range[0], eps_range[1], eps_steps)
        min_samples_values = list(range(min_samples_range[0], min_samples_range[1] + 1))

        results = {
            'eps_values': eps_values.tolist(),
            'min_samples_values': min_samples_values,
            'n_clusters_matrix': [],
            'noise_ratio_matrix': [],
            'silhouette_matrix': []
        }

        print("搜索DBSCAN最优参数...")

        for i, eps in enumerate(eps_values):
            n_clusters_row = []
            noise_ratio_row = []
            silhouette_row = []

            for min_samples in min_samples_values:
                print(f"  测试 eps={eps:.3f}, min_samples={min_samples}")

                dbscan_temp = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                labels_temp = dbscan_temp.fit_predict(features)

                # 计算聚类数量和噪声比例
                unique_labels = np.unique(labels_temp)
                n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
                noise_ratio = np.sum(labels_temp == -1) / len(labels_temp)

                n_clusters_row.append(n_clusters_found)
                noise_ratio_row.append(noise_ratio)

                # 计算轮廓系数（只在没有噪声且有多个聚类时）
                if n_clusters_found > 1 and noise_ratio < 0.5:
                    valid_mask = labels_temp != -1
                    if np.sum(valid_mask) > 1 and len(np.unique(labels_temp[valid_mask])) > 1:
                        try:
                            silhouette_avg = silhouette_score(features[valid_mask], labels_temp[valid_mask])
                            silhouette_row.append(silhouette_avg)
                        except:
                            silhouette_row.append(0.0)
                    else:
                        silhouette_row.append(0.0)
                else:
                    silhouette_row.append(0.0)

            results['n_clusters_matrix'].append(n_clusters_row)
            results['noise_ratio_matrix'].append(noise_ratio_row)
            results['silhouette_matrix'].append(silhouette_row)

        if plot:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

            # 聚类数量热力图
            im1 = ax1.imshow(results['n_clusters_matrix'], cmap='viridis', aspect='auto')
            ax1.set_xticks(range(len(min_samples_values)))
            ax1.set_xticklabels(min_samples_values)
            ax1.set_yticks(range(len(eps_values)))
            ax1.set_yticklabels([f'{eps:.2f}' for eps in eps_values])
            ax1.set_xlabel('min_samples')
            ax1.set_ylabel('eps')
            ax1.set_title('Number of Clusters Found')
            plt.colorbar(im1, ax=ax1)

            # 噪声比例热力图
            im2 = ax2.imshow(results['noise_ratio_matrix'], cmap='Reds', aspect='auto')
            ax2.set_xticks(range(len(min_samples_values)))
            ax2.set_xticklabels(min_samples_values)
            ax2.set_yticks(range(len(eps_values)))
            ax2.set_yticklabels([f'{eps:.2f}' for eps in eps_values])
            ax2.set_xlabel('min_samples')
            ax2.set_ylabel('eps')
            ax2.set_title('Noise Ratio')
            plt.colorbar(im2, ax=ax2)

            # 轮廓系数热力图
            im3 = ax3.imshow(results['silhouette_matrix'], cmap='Blues', aspect='auto')
            ax3.set_xticks(range(len(min_samples_values)))
            ax3.set_xticklabels(min_samples_values)
            ax3.set_yticks(range(len(eps_values)))
            ax3.set_yticklabels([f'{eps:.2f}' for eps in eps_values])
            ax3.set_xlabel('min_samples')
            ax3.set_ylabel('eps')
            ax3.set_title('Silhouette Score')
            plt.colorbar(im3, ax=ax3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"DBSCAN参数分析图已保存: {save_path}")

            plt.show()

        return results


if __name__ == "__main__":
    # 测试DBSCAN聚类分析
    from sklearn.datasets import make_blobs

    # 创建模拟数据
    X, y = make_blobs(n_samples=1000, centers=10, n_features=128, random_state=42, cluster_std=1.5)

    # 创建DBSCAN聚类分析器
    analyzer = ClusteringAnalysis(eps=0.8, min_samples=5)

    # 执行聚类
    results = analyzer.fit_dbscan(X)

    # 评估聚类质量
    metrics = analyzer.evaluate_clustering(X, y)

    # 分析聚类分布
    distribution_df = analyzer.analyze_cluster_distribution(y)
    print("\n聚类分布分析:")
    print(distribution_df.head(10))

    # 寻找最优参数（可选）
    # optimal_results = analyzer.find_optimal_eps(X, eps_range=(0.1, 2.0), eps_steps=10)