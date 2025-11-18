#!/usr/bin/env python3
"""
测试DBSCAN聚类分析脚本
使用默认参数运行完整的自监督DBSCAN聚类流程
"""

import sys
from pathlib import Path
import subprocess

# 设置项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("开始测试自监督DBSCAN聚类分析...")

    # 设置参数
    script_path = PROJECT_ROOT / "src" / "clustering_analysis" / "selfsupervised_clustering.py"

    # DBSCAN参数
    eps = 0.5
    min_samples = 5

    cmd = [
        "python", str(script_path),
        "--data_root", "data/LFM_dataset/data_noise_50",
        "--arch_type", "ca1d",
        "--train_ratio", "0.8",
        "--batch_size", "256",
        "--epochs", "50",
        "--lr", "1e-4",
        "--eps", str(eps),
        "--min_samples", str(min_samples),
        "--random_state", "42",
        "--output_dir", "dbscan_clustering_results",
        "--model_save_path", "pretrained_models/dbscan_backbone.pth",
        "--umap_neighbors", "15",
        "--umap_min_dist", "0.1",
        "--save_plots"
    ]

    print(f"执行命令: {' '.join(cmd)}")
    print(f"DBSCAN参数: eps={eps}, min_samples={min_samples}")

    try:
        # 运行聚类分析
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ DBSCAN聚类分析完成!")
        print("输出:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print("❌ DBSCAN聚类分析失败!")
        print("错误信息:")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()