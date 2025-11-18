
# 真正的开集分类分析报告

## 数据配置
- 已知类别数: 7 (类1-7)
- 未知类别数: 3 (类8-10)

## 训练数据
- 已知类样本: 28000 (每类4000)
- 未知类样本: 0 (每类40，无标签)
- 总训练样本: 28000

## 测试数据
- 已知类样本: 7000 (每类1000)
- 未知类样本: 3000 (每类1000，无标签)
- 总测试样本: 10000 (10*1000条数据)

## 预训练配置
- 骨干网络: ca1d
- 特征维度: 128
- 预训练轮数: 50
- 学习率: 1e-4

## 聚类结果
- 发现聚类数: 1
- 噪声点数: 1 (0.0%)

## 开集识别性能
- 已知类ARI: 0.9998
- 已知类NMI: 0.9997
- 未知类发现聚类数: 3
- 未知类噪声比例: 0.0000

## 文件输出
- 特征文件: true_open_set_features.npy
- UMAP嵌入: true_open_set_embeddings.npy
- 真实标签: true_open_set_true_labels.npy
- 聚类标签: true_open_set_cluster_labels.npy
- 可视化图: true_open_set_umap_analysis.png
- 预训练模型: true_open_set_pretrained_backbone.pth
