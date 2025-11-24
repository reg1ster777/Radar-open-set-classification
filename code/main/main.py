#!/usr/bin/env python3
"""项目入口，串联数据构建、预训练、监督微调与开集评估。"""

import sys
import warnings

import torch

import config
warnings.filterwarnings("ignore")

# 确保可以导入 src.* 模块
sys.path.insert(0, str(config.PROJECT_ROOT))

from data_utils import (
    create_open_set_dataset,
    ensure_channel_dim,
    build_supervised_loader,
)
from reconstruction import (
    train_feature_extractor,
    visualize_reconstruction_per_class,
)
from supervised_training import train_supervised_classifier, evaluate_classifier
from open_set_eval import evaluate_open_set


def main():
    print("=" * 80)
    print("data1 数据集 | 特征提取 + 重构 + 监督分类 + 开集评估")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    dataset = create_open_set_dataset()

    train_signals = ensure_channel_dim(dataset['train_known_data'])
    train_labels = dataset['train_known_labels']
    test_known_signals = ensure_channel_dim(dataset['test_known_data'])
    test_known_labels = dataset['test_known_labels']

    print("\n=== 3. 重构式预训练 + 监督对比 ===")
    pretrainer = train_feature_extractor(train_signals, train_labels, device)

    visualize_reconstruction_per_class(
        train_signals,
        train_labels,
        pretrainer,
        device,
        save_dir=config.RECON_DIR,
        samples_per_class=10,
    )

    print("\n=== 4. 开集评估 ===")
    dataset['train_known_data'] = train_signals  # 保留补齐通道后的数据
    dataset['test_known_data'] = test_known_signals
    dataset['test_unknown_data'] = ensure_channel_dim(dataset['test_unknown_data'])
    evaluate_open_set(pretrainer, device, dataset)


if __name__ == "__main__":
    main()
