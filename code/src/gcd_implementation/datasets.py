import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from scipy.io import loadmat
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

# 导入现有数据加载工具函数
from src.solver.dataset import _extract_label_from_name, _read_mat_array, _stratified_split


def mask_signal(signal: torch.Tensor, masked_ratio: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对信号进行随机掩码。
    Args:
        signal: 输入信号，形状为 [batch_size, channels, length]。
        masked_ratio: 掩码比例。
    Returns:
        masked_signal: 被掩码后的信号。
        masked_patches_values: 被掩码部分的原始值。
        masked_indices: 被掩码部分的索引 (对于批次中的所有信号相同)。
    """
    batch_size, channels, length = signal.shape
    num_masked_elements = int(length * masked_ratio)
    
    # 随机选择要掩码的索引，并将其移动到设备上
    masked_indices = torch.randperm(length)[:num_masked_elements].to(signal.device)
    
    masked_signal = signal.clone()
    
    # 使用高级索引从所有批次和通道中获取被掩码部分的原始值
    # masked_patches_values 将是 [batch_size, channels, num_masked_elements]
    masked_patches_values = signal[:, :, masked_indices].clone() 
    
    # 将被掩码部分置零
    masked_signal[:, :, masked_indices] = 0.0
    
    return masked_signal, masked_patches_values, masked_indices


def augment_weak(signal: torch.Tensor) -> torch.Tensor:
    """
    弱数据增强：添加少量高斯噪声。
    """
    noise = torch.randn_like(signal) * 0.01  # 假设噪声标准差为0.01
    return signal + noise


def augment_strong(signal: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    """
    强数据增强：添加更多高斯噪声（不再使用掩码）。
    Args:
        signal: 输入信号
        noise_std: 噪声标准差
    """
    noise = torch.randn_like(signal) * noise_std
    return signal + noise


class GCDDataset(Dataset):
    """
    用于GCD的雷达数据集，支持加载所有数据并提供原始信号、标签和索引。
    """
    def __init__(
        self,
        root: Union[str, Path],
        mat_key: Optional[str] = "all_data",
        samples_axis: int = 0,
        dtype: torch.dtype = torch.float32,
        known_class_labels: Optional[Sequence[int]] = None,
        unlabeled_class_labels: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"not a directory: {self.root}")

        files = sorted(self.root.glob("**/*.mat")) # 递归查找所有.mat文件
        if not files:
            raise FileNotFoundError(f"no .mat files under: {self.root}")

        file_info: List[Tuple[Path, int]] = []
        for f in files:
            lab = _extract_label_from_name(f)
            if lab is None:
                # 如果文件名不符合 xxx_<label>.mat 格式，尝试从父目录名获取标签
                # 这是一个临时的处理，可能需要更健壮的逻辑
                try:
                    lab = int(f.parent.name.split('_')[-1]) # 假设目录名是 LFM_dataset/data_noise_50
                except ValueError:
                    raise ValueError(
                        f"file name does not match 'xxx_<label>.mat' and cannot infer label from parent: {f.name}"
                    )
            file_info.append((f, lab))
        unique_labels = sorted({lab for _, lab in file_info})

        # 过滤：只保留指定的“原始标签”
        kept_raw_labels = set()
        if known_class_labels is not None:
            kept_raw_labels.update(int(x) for x in known_class_labels)
        if unlabeled_class_labels is not None:
            kept_raw_labels.update(int(x) for x in unlabeled_class_labels)
        
        if not kept_raw_labels:
            kept_raw_labels.update(unique_labels) # 如果没有指定，则保留所有标签

        missing = kept_raw_labels - set(unique_labels)
        if missing:
            raise ValueError(f"labels not found in folder: {sorted(missing)}")

        # 建立原始标签 -> 新标签（1..K）的映射
        kept_sorted = sorted(kept_raw_labels)
        self.label_map: Dict[int, int] = {
            old: new + 1 for new, old in enumerate(kept_sorted)}

        data_list: List[torch.Tensor] = []
        label_list: List[torch.Tensor] = []
        original_indices: List[int] = [] # 记录原始索引

        for idx, (path, raw_lab) in enumerate(file_info):
            if raw_lab not in kept_raw_labels:
                continue

            arr = _read_mat_array(path, mat_key=mat_key)
            # 将样本维度移到前面并展平
            arr = np.moveaxis(arr, samples_axis, 0)
            # arr = arr.reshape(arr.shape[0], -1)  # [n_i, D] - 暂时不展平，保留原始信号结构
            # 确保信号是 [channels, length] 或 [length]
            if arr.ndim == 2: # 如果是 [samples, length] 假设 channels=1
                arr = arr[:, np.newaxis, :]
            elif arr.ndim == 1: # 如果是 [length] 假设 samples=1, channels=1
                arr = arr[np.newaxis, np.newaxis, :]
            
            x = torch.from_numpy(arr).to(dtype)
            y_new = self.label_map[raw_lab]
            y = torch.full((x.shape[0],), y_new, dtype=torch.long)

            data_list.append(x)
            label_list.append(y)
            original_indices.extend([raw_lab] * x.shape[0]) # 每个样本对应原始标签

        if not data_list:
            raise RuntimeError(
                "no samples collected (check known_class_labels filter).")

        self.data = torch.cat(data_list, dim=0)     # [N, C, L]
        self.labels = torch.cat(label_list, dim=0)  # [N]
        self.num_classes = len(kept_sorted)
        self.original_indices = original_indices
        self.original_signal_length = self.data.shape[2] # 假设信号长度在第三维
        self.input_channels = self.data.shape[1] # 假设通道数在第二维

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # 返回原始信号，标签，以及原始文件索引
        return self.data[idx], self.labels[idx], self.original_indices[idx]

    def __repr__(self) -> str:
        return (f"GCDDataset(root={self.root.name}, N={len(self)}, "
                f"signal_shape={self.data.shape[1:]}, K={self.num_classes})")


# 辅助函数，用于创建D_labeled和D_unlabeled的索引
def get_gcd_split_indices(
    dataset: GCDDataset,
    known_class_labels: Sequence[int],
    unlabeled_class_labels: Sequence[int],
    labeled_ratio: float = 0.5,  # 已知类别中有多少比例放入有标签数据集
) -> Tuple[List[int], List[int]]:
    """
    根据提供的已知类和无标签类，从GCDDataset中获取D_labeled和D_unlabeled的索引。

    GCD标准设定：
    - 已知类别：部分样本放入有标签数据集，剩余放入无标签数据集
    - 未知类别：全部放入无标签数据集

    Args:
        dataset: GCDDataset实例
        known_class_labels: 已知类别的原始标签列表
        unlabeled_class_labels: 无标签数据集中可能包含的类别标签列表
        labeled_ratio: 已知类别中分配给有标签数据集的比例

    Returns:
        labeled_indices: 有标签数据集的索引
        unlabeled_indices: 无标签数据集的索引
    """
    import random

    labeled_indices = []
    unlabeled_indices = []

    # 按类别组织样本索引
    class_to_indices = {}
    for i, (signal, label, original_idx) in enumerate(dataset):
        # 将数据集中的重映射标签转换回原始标签进行判断
        original_label = next(key for key, value in dataset.label_map.items() if value == label.item())

        if original_label not in class_to_indices:
            class_to_indices[original_label] = []
        class_to_indices[original_label].append(i)

    # 处理每个类别
    for original_label, indices in class_to_indices.items():
        if original_label in known_class_labels:
            # 已知类别：按比例分割
            random.shuffle(indices)  # 随机打乱
            split_point = int(len(indices) * labeled_ratio)
            labeled_indices.extend(indices[:split_point])
            unlabeled_indices.extend(indices[split_point:])
        elif original_label in unlabeled_class_labels:
            # 未知类别：全部放入无标签数据集
            unlabeled_indices.extend(indices)
        # 如果类别既不在已知类也不在无标签类中，则忽略

    return labeled_indices, unlabeled_indices