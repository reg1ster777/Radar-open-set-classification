import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Subset, TensorDataset
import os
import random
import sys

# Allow running the script directly by adding the repo root to sys.path.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.gcd_implementation.datasets import GCDDataset, get_gcd_split_indices
from src.gcd_implementation.gcd_solver import GCDPretrainer, LabeledDataPretrainer, GCDFinetuner, UnknownClassFinetuner, GCDEvaluator
from src.gcd_implementation.models import get_backbone
from src.gcd_implementation.ground_truth_splitter import GroundTruthSplitter
from src.gcd_implementation.alternating_finetuner import AlternatingFinetuner

def main():
    parser = argparse.ArgumentParser(description="Generalized Category Discovery (GCD) Training")
    parser.add_argument("--arch_type", type=str, default="ca1d", choices=["ca1d", "hybrid", "mamba"], help="Backbone architecture type")
    parser.add_argument("--data_root", type=str, default="./data/LFM_dataset", help="Root directory of the dataset")
    parser.add_argument("--known_class_count", type=int, default=7, help="Number of known classes (K)")
    parser.add_argument("--unknown_class_count", type=int, default=3, help="Number of unknown classes (M)")
    parser.add_argument("--epochs_pretrain", type=int, default=10, help="Number of epochs for pre-training")
    parser.add_argument("--epochs_finetune", type=int, default=10, help="Number of epochs for GCD fine-tuning")
    parser.add_argument("--lr_pretrain", type=float, default=1e-4, help="Learning rate for pre-training")
    parser.add_argument("--lr_finetune", type=float, default=1e-3, help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--compression_ratio", type=float, default=0.25, help="Compression ratio for reconstruction head")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone during fine-tuning (default behavior)")
    parser.add_argument("--unfreeze_backbone", action="store_true", help="Unfreeze backbone during fine-tuning (train both backbone and classification head)")

    # 穿插式微调参数
    parser.add_argument("--alternating_finetune", action="store_true", help="Use alternating fine-tuning (labeled + unknown together)")
    parser.add_argument("--epochs_alternating", type=int, default=10, help="Number of epochs for alternating fine-tuning")
    parser.add_argument("--lr_alternating", type=float, default=1e-3, help="Learning rate for alternating fine-tuning")
    parser.add_argument("--labeled_weight", type=float, default=1.0, help="Weight for labeled loss in alternating training")
    parser.add_argument("--unknown_weight", type=float, default=0.5, help="Weight for unknown loss in alternating training")
    parser.add_argument("--unknown_ratio", type=float, default=0.3, help="Target ratio of unknown samples in mixed batches")
    parser.add_argument("--distribution_sharpness", type=float, default=5.0, help="Sharpness parameter for unknown distribution targeting")
    parser.add_argument("--entropy_weight", type=float, default=0.1, help="Weight for entropy regularization in alternating training")

    # 传统分离式微调参数（向后兼容）
    parser.add_argument("--epochs_unknown_finetune", type=int, default=15, help="Number of epochs for unknown class fine-tuning")
    parser.add_argument("--lr_unknown_finetune", type=float, default=5e-4, help="Learning rate for unknown class fine-tuning")
    parser.add_argument("--freeze_backbone_unknown", action="store_true", help="Freeze backbone during unknown class fine-tuning")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # 处理backbone冻结逻辑：默认冻结，除非明确指定解冻
    if args.unfreeze_backbone:
        args.freeze_backbone = False
    else:
        args.freeze_backbone = True  # 默认冻结

    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 数据加载与准备 ---
    # 假设data_root是到LFM_dataset的路径，但实际数据在LFM_dataset/data_noise_50
    # 需要调整data_root以匹配GCDDataset的递归查找逻辑
    # 或者直接传入LFM_dataset的父目录，让GCDDataset去查找
    # 这里为了简化，假设data_root直接指向包含.mat文件的目录，例如 LFM_dataset/data_noise_50
    # 如果你的数据结构是 data/LFM_dataset/data_noise_50/*.mat，那么data_root应该设置为 data/LFM_dataset
    # GCDDataset会递归查找
    
    # 示例：假设data_root是项目根目录下的data文件夹
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent # 从 code/gcd_implementation 到项目根目录
    actual_data_root = project_root / Path(args.data_root)

    print(f"Loading data from: {actual_data_root}")
    # GCDDataset会从文件名提取标签，并重映射
    # 这里需要提供所有可能出现的原始标签，以便GCDDataset正确构建label_map
    # 假设原始标签从1开始，且是连续的
    all_possible_raw_labels = list(range(1, args.known_class_count + args.unknown_class_count + 10)) # 预留一些空间

    full_dataset = GCDDataset(
        root=actual_data_root,
        known_class_labels=list(range(1, args.known_class_count + 1)), # 假设已知类原始标签是1到K
        unlabeled_class_labels=list(range(1, args.known_class_count + args.unknown_class_count + 1)), # 假设无标签数据包含所有K+M类
    )

    original_signal_length = full_dataset.original_signal_length
    input_channels = full_dataset.input_channels
    num_total_classes = full_dataset.num_classes # 这是重映射后的总类别数

    # --- 第一步：基于真实标签的数据集分析 ---
    print("\n=== 第一步：基于真实标签的数据集分析 ===")
    splitter = GroundTruthSplitter(known_class_count=args.known_class_count)

    # 分析数据集分布
    analysis = splitter.analyze_dataset(full_dataset)

    # 基于真实标签分割数据集
    labeled_dataset, unlabeled_dataset, unknown_indices, known_indices = splitter.split_dataset_by_labels(
        full_dataset, labeled_ratio=0.8
    )

    # 创建未知类别数据集
    unknown_dataset = splitter.create_unknown_dataset(full_dataset, unknown_indices)

    # 保存分割结果
    model_save_dir = project_root / "model" / f"{args.arch_type}_gcd"
    splitter.save_split_results(
        [labeled_dataset.indices], unknown_indices, known_indices, model_save_dir
    )

    # 对于预训练，使用所有数据
    pretrain_dataset = full_dataset

    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)  # 第四阶段需要
    unknown_dataloader = DataLoader(unknown_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)  # 穿插式微调需要
    eval_dataloader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- 2. 阶段一：自监督预训练 ---
    print("\n--- Phase 1: Self-supervised Pre-training ---")
    _, feature_dim = get_backbone(args.arch_type, input_channels=input_channels)
    pretrainer = GCDPretrainer(
        arch_type=args.arch_type,
        original_signal_length=original_signal_length,
        input_channels=input_channels,
        feature_dim=feature_dim,
        device=device,
        lr=args.lr_pretrain,
        compression_ratio=args.compression_ratio,
    )

    for epoch in range(args.epochs_pretrain):
        loss = pretrainer.train_epoch(pretrain_dataloader)
        print(f"Pretrain Epoch {epoch+1}/{args.epochs_pretrain}, Loss: {loss:.4f}")
    
    # 保存预训练的Backbone
    model_save_dir = project_root / "model" / f"{args.arch_type}_gcd"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    pretrain_backbone_path = model_save_dir / f"pretrain_backbone_{args.arch_type}.pth"
    pretrainer.save_backbone(pretrain_backbone_path)

    # --- 2. 阶段二：直接GCD微调 ---
    print("\n--- Phase 2: GCD Fine-tuning ---")
    # 加载预训练的backbone
    finetune_backbone, _ = get_backbone(args.arch_type, input_channels=input_channels)
    pretrain_checkpoint = torch.load(pretrain_backbone_path, map_location=device)
    if isinstance(pretrain_checkpoint, dict) and "backbone_state_dict" in pretrain_checkpoint:
        finetune_backbone.load_state_dict(pretrain_checkpoint["backbone_state_dict"])
    else:
        finetune_backbone.load_state_dict(pretrain_checkpoint)

    # 创建简化的GCD微调器（不需要重构头）
    finetuner = GCDFinetuner(
        backbone=finetune_backbone,
        labeled_reconstruction_head=None,  # 不再需要重构头
        feature_dim=feature_dim,
        num_known_classes=args.known_class_count, # 这里是原始的K
        num_total_classes=num_total_classes, # 这是重映射后的K+M
        device=device,
        lr=args.lr_finetune,
        freeze_backbone=args.freeze_backbone,
        reconstruction_error_threshold=0.1,  # 设置一个默认值，实际不会使用
    )

    for epoch in range(args.epochs_finetune):
        loss = finetuner.train_epoch(labeled_dataloader)
        print(f"Finetune Epoch {epoch+1}/{args.epochs_finetune}, Loss: {loss:.4f}")

    # 保存微调后的模型
    finetuned_model_path = model_save_dir / f"finetuned_gcd_model_{args.arch_type}.pth"
    finetuner.save_model(finetuned_model_path)

    if args.alternating_finetune:
        # --- 3a. 穿插式微调 ---
        print("\n--- Phase 3: Alternating Fine-tuning ---")

        # 创建穿插式微调器
        alternating_finetuner = AlternatingFinetuner(
            backbone=finetune_backbone,
            feature_dim=feature_dim,
            num_known_classes=args.known_class_count,
            num_total_classes=num_total_classes,
            device=device,
            lr=args.lr_alternating,
            freeze_backbone=args.freeze_backbone,
            labeled_weight=args.labeled_weight,
            unknown_weight=args.unknown_weight,
            unknown_ratio=args.unknown_ratio,
            distribution_sharpness=args.distribution_sharpness,
            entropy_weight=args.entropy_weight,
        )

        # 穿插式微调
        print(f"开始穿插式微调，训练 {args.epochs_alternating} 个epoch...")
        print(f"  有标签损失权重: {args.labeled_weight}")
        print(f"  未知损失权重: {args.unknown_weight}")
        print(f"  目标未知样本比例: {args.unknown_ratio}")

        for epoch in range(args.epochs_alternating):
            metrics = alternating_finetuner.train_epoch_alternating(labeled_dataloader, unknown_dataloader)
            print(f"Alternating Epoch {epoch+1}/{args.epochs_alternating}, Total Loss: {metrics['total_loss']:.4f}")

        # 评估穿插式微调效果
        print("\n评估穿插式微调效果:")
        eval_metrics = alternating_finetuner.evaluate_alternating(labeled_dataloader, unknown_dataloader)

        # 保存最终模型
        final_model_path = model_save_dir / f"alternating_finetuned_model_{args.arch_type}.pth"
        alternating_finetuner.save_model(final_model_path)

        # 最终评估
        print("\n--- Phase 4: Final Evaluation ---")
        final_evaluator = GCDEvaluator(
            backbone=finetune_backbone,
            classification_head=alternating_finetuner.classification_head,
            device=device,
        )
        _, _ = final_evaluator.evaluate(eval_dataloader, num_total_classes)

    else:
        # --- 3b. 传统分离式微调 ---
        print("\n--- Phase 3: Labeled Fine-tuning ---")

        # 创建传统有标签微调器
        finetuner = GCDFinetuner(
            backbone=finetune_backbone,
            labeled_reconstruction_head=None,
            feature_dim=feature_dim,
            num_known_classes=args.known_class_count,
            num_total_classes=num_total_classes,
            device=device,
            lr=args.lr_finetune,
            freeze_backbone=args.freeze_backbone,
            reconstruction_error_threshold=0.1,
        )

        # 有标签微调
        for epoch in range(args.epochs_finetune):
            loss = finetuner.train_epoch(labeled_dataloader)
            print(f"Labeled Finetune Epoch {epoch+1}/{args.epochs_finetune}, Loss: {loss:.4f}")

        # 保存微调后的模型
        finetuned_model_path = model_save_dir / f"labeled_finetuned_model_{args.arch_type}.pth"
        finetuner.save_model(finetuned_model_path)

        # --- 4b. 未知类别微调 ---
        print("\n--- Phase 4: Unknown Class Fine-tuning ---")

        # 直接使用第一步中识别的未知类别数据
        if len(unknown_indices) == 0:
            print("警告：未找到未知类别样本，跳过第四阶段微调")
        else:
            print(f"使用第一步识别的未知类别数据: {len(unknown_indices)} 样本")
            unknown_dataloader = DataLoader(unknown_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

            print(f"使用基于真实标签的未知类别数据集")

            # 创建未知类别微调器
            unknown_finetuner = UnknownClassFinetuner(
                backbone=finetune_backbone,
                classification_head=finetuner.classification_head,
                known_class_count=args.known_class_count,
                num_total_classes=num_total_classes,
                device=device,
                lr=args.lr_unknown_finetune,
                freeze_backbone=args.freeze_backbone_unknown,
                entropy_weight=args.entropy_weight,
                distribution_sharpness=args.distribution_sharpness,
            )

            # 未知类别微调
            print(f"开始未知类别微调，训练 {args.epochs_unknown_finetune} 个epoch...")
            for epoch in range(args.epochs_unknown_finetune):
                loss = unknown_finetuner.train_epoch(unknown_dataloader)
                print(f"Unknown Finetune Epoch {epoch+1}/{args.epochs_unknown_finetune}, Loss: {loss:.4f}")

            # 评估未知类别微调效果
            print("\n评估未知类别微调效果:")
            unknown_metrics = unknown_finetuner.evaluate_unknown_predictions(unknown_dataloader)
            print(f"  未知样本总数: {unknown_metrics['total_samples']}")
            print(f"  预测分布: {unknown_metrics['prediction_distribution']}")
            print(f"  平均置信度: {unknown_metrics['avg_confidence']:.4f}")
            print(f"  平均熵: {unknown_metrics['avg_entropy']:.4f}")

            # 保存最终模型
            final_model_path = model_save_dir / f"final_gcd_model_with_unknown_{args.arch_type}.pth"
            unknown_finetuner.save_model(final_model_path)

            # 更新evaluator使用最终模型
            final_evaluator = GCDEvaluator(
                backbone=finetune_backbone,
                classification_head=finetuner.classification_head,
                device=device,
            )

            # 最终评估
            print("\n--- Phase 5: Final Evaluation after Unknown Fine-tuning ---")
            _, _ = final_evaluator.evaluate(eval_dataloader, num_total_classes)

    print("Script finished successfully.")

if __name__ == "__main__":
    main()
