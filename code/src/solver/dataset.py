# dataset.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from scipy.io import loadmat


# --------------------------- 工具函数 ---------------------------

# 支持带或不带下划线的命名，例如 "class_1.mat" 或 "data1.mat"
_LABEL_PAT = re.compile(r".*?[_]?(\d+)\.mat$", flags=re.IGNORECASE)


def _extract_label_from_name(path: Path) -> Optional[int]:
    """
    从文件名中提取末尾的数字作为类别标签，例如 'xxx_12.mat' -> 12
    """
    m = _LABEL_PAT.match(path.name)
    return int(m.group(1)) if m else None


def _read_mat_array(path: Path, mat_key: Optional[str]) -> np.ndarray:
    """
    读取 .mat 文件，优先取 mat_key；否则自动挑一个合适的数组。
    返回 float32 numpy 数组（至少二维）。
    """
    mat = loadmat(path.as_posix())
    if mat_key is not None:
        if mat_key not in mat:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(
                f"[{path.name}] key '{mat_key}' not found. available: {keys}"
            )
        arr = mat[mat_key]
    else:
        candidates = [
            v for k, v in mat.items()
            if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim >= 2
        ]
        if not candidates:
            raise KeyError(f"[{path.name}] no suitable ndarray found.")
        arr = candidates[0]

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"[{path.name}] array must be >=2D, got {arr.shape}")
    return arr


def _stratified_split(
    labels: np.ndarray,
    train_ratio: float,
    random_state: int = 42,
    min_train: int = 1,
    min_test: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按类别分层切分索引，保证每类尽量都有 train/test。
    """
    assert 0.0 < train_ratio < 1.0
    rng = np.random.RandomState(random_state)
    tr, te = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        # 为避免极端小样本类被切空，做下限保护
        if n <= (min_train + min_test):
            n_tr = min(n, max(min_train, int(round(n * train_ratio))))
        else:
            n_tr = max(min_train, int(round(n * train_ratio)))
            n_tr = min(n_tr, n - min_test)
        tr.extend(idx[:n_tr].tolist())
        te.extend(idx[n_tr:].tolist())
    return np.array(tr, dtype=np.int64), np.array(te, dtype=np.int64)


def _stratified_split_fixed_test(
    labels: np.ndarray,
    test_per_class: int,
    random_state: int = 42,
    min_train: int = 1,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按类别分层切分索引，每个类别固定测试集样本数量。
    """
    assert test_per_class > 0
    tr, te = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if shuffle:
            rng = np.random.RandomState(random_state + int(cls))
            rng.shuffle(idx)
        n = len(idx)

        # 确保有足够的样本进行切分
        if n <= test_per_class:
            n_te = min(n, test_per_class)
            n_tr = n - n_te
        else:
            n_te = test_per_class
            n_tr = n - n_te

        tr.extend(idx[:n_tr].tolist())
        te.extend(idx[n_tr:n_tr+n_te].tolist())
    return np.array(tr, dtype=np.int64), np.array(te, dtype=np.int64)


# --------------------------- 数据集类 ---------------------------


class LabelOverride(Dataset):
    """把底层数据集的标签覆写为固定值（默认 -1），并统一成 long Tensor。"""

    def __init__(self, base_ds: Dataset, label: int = -1):
        self.base = base_ds
        self.label = int(label)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]          # RadarDataset 已返回 Tensor
        if not torch.is_tensor(x):     # 兜底：确保是 Tensor
            x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.tensor(self.label, dtype=torch.long)   # 关键：标签统一成 LongTensor
        return x, y


class RadarDataset(Dataset):
    """
    雷达数据集（一次性载入内存版）。

    约定：
      - 目录下若干 .mat 文件，每个文件名形如 'xxx_<label>.mat'，其中 <label> 为该文件的类别号；
      - 每个 .mat 文件中的样本都属于同一类；
      - 数组中样本沿 samples_axis 堆叠，其他维度会被展平为特征。

    重要属性：
      - data:   [N, D] float32
      - labels: [N]    long（已重映射为 0..K-1）
      - label_map:  原始标签 -> 新标签 的映射
      - class_to_files: 新标签 -> 文件名列表（可能多个文件属于同一原始标签时）
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        # 传入“要保留的原始标签集合”；None表示全部
        known_class_labels: Optional[Sequence[int]] = None,
        mat_key: Optional[str] = "all_data",
        samples_axis: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"not a directory: {self.root}")

        files = sorted(self.root.glob("*.mat"))
        if not files:
            raise FileNotFoundError(f"no .mat files under: {self.root}")

        # 统计目录中“可用的原始标签”
        file_info: List[Tuple[Path, int]] = []
        for f in files:
            lab = _extract_label_from_name(f)
            if lab is None:
                raise ValueError(
                    f"file name does not match 'xxx_<label>.mat': {f.name}")
            file_info.append((f, lab))
        unique_labels = sorted({lab for _, lab in file_info})

        # 过滤：只保留指定的“原始标签”
        if known_class_labels is None:
            kept_labels = set(unique_labels)
        else:
            kept_labels = set(int(x) for x in known_class_labels)
            missing = kept_labels - set(unique_labels)
            if missing:
                raise ValueError(
                    f"labels not found in folder: {sorted(missing)}")

        # 建立原始标签 -> 新标签（0..K-1）的映射
        kept_sorted = sorted(kept_labels)
        self.label_map: Dict[int, int] = {
            old: new for new, old in enumerate(kept_sorted)}

        # 读取 & 拼接
        data_list: List[Tensor] = []
        label_list: List[Tensor] = []
        class_to_files: Dict[int, List[str]] = {}

        for path, raw_lab in file_info:
            if raw_lab not in kept_labels:
                continue

            arr = _read_mat_array(path, mat_key=mat_key)
            # 将样本维度移到前面并展平
            arr = np.moveaxis(arr, samples_axis, 0)
            arr = arr.reshape(arr.shape[0], -1)  # [n_i, D]

            x = torch.from_numpy(arr).to(dtype)
            y_new = self.label_map[raw_lab]
            y = torch.full((x.shape[0],), y_new, dtype=torch.long)

            data_list.append(x)
            label_list.append(y)
            class_to_files.setdefault(y_new, []).append(path.name)

        if not data_list:
            raise RuntimeError(
                "no samples collected (check known_class_labels filter).")

        self.data = torch.cat(data_list, dim=0)     # [N, D]
        self.labels = torch.cat(label_list, dim=0)  # [N]
        self.num_classes = len(kept_sorted)
        self.class_to_files = class_to_files

    # ---- Dataset 接口 ----
    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx], self.labels[idx]

    def __repr__(self) -> str:
        return (f"RadarDataset(root={self.root.name}, N={len(self)}, D={self.data.shape[1]}, "
                f"K={self.num_classes})")


# --------------------------- 入口函数 ---------------------------

def load_radar_known(
    root: Union[str, Path],
    known_class_count: int,
    train_ratio: float,
    *,
    mat_key: Optional[str] = "all_data",
    samples_axis: int = 0,
    batch_size: int = 128,
    shuffle_train: bool = True,
    shuffle_test: bool = False,
    random_state: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    as_loader: bool = True,
) -> Union[
    Tuple[DataLoader, DataLoader],
    Tuple[Dataset, Dataset]
]:
    """
    加载所有已知类数据

    载入文件夹内所有 .mat：
      1) 自动从文件名 'xxx_<label>.mat' 解析原始标签；
      2) 校验 known_class_count <= 可用 .mat 文件数（或去重后的标签数）；
      3) 选择“排序后前 K=known_class_count 个原始标签”并保留；
      4) 构造 RadarDataset（将保留的原始标签重映射为 0..K-1）；
      5) 分层切分 train/test 并返回（可选 DataLoader）。

    返回：(train, test)
    """
    root = Path(root)
    files = sorted(root.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"no .mat files under: {root}")

    # 提取目录中的“原始标签集合”
    raw_labels = []
    for f in files:
        lab = _extract_label_from_name(f)
        if lab is None:
            raise ValueError(
                f"file name does not match 'xxx_<label>.mat': {f.name}")
        raw_labels.append(lab)

    unique_labels = sorted(set(raw_labels))

    # 既满足“文件数”也满足“不同标签数”的语义（严格一些）
    if known_class_count > len(files) or known_class_count > len(unique_labels):
        raise ValueError(
            f"known_class_count={known_class_count} must be <= #files({len(files)}) "
            f"and <= #unique_labels({len(unique_labels)})."
        )
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")

    # 选择前 K 个原始标签作为“已知类”
    kept_raw_labels = unique_labels[:known_class_count]

    # 构造数据集（仅保留所选标签，并重映射为 0..K-1）
    ds_full = RadarDataset(
        root,
        known_class_labels=kept_raw_labels,
        mat_key=mat_key,
        samples_axis=samples_axis,
    )

    # 分层切分
    tr_idx, te_idx = _stratified_split(
        labels=ds_full.labels.numpy(),
        train_ratio=train_ratio,
        random_state=random_state,
    )
    ds_train = Subset(ds_full, tr_idx)
    ds_test = Subset(ds_full, te_idx)

    if not as_loader:
        return ds_train, ds_test
    else:
        tr_loader = DataLoader(
            ds_train, batch_size=batch_size, shuffle=shuffle_train,
            num_workers=num_workers, pin_memory=pin_memory
        )
        te_loader = DataLoader(
            ds_test, batch_size=batch_size, shuffle=shuffle_test,
            num_workers=num_workers, pin_memory=pin_memory
        )
        return tr_loader, te_loader


def load_radar_known_fixed_test(
    root: Union[str, Path],
    known_class_count: int,
    test_per_class: int,
    *,
    mat_key: Optional[str] = "all_data",
    samples_axis: int = 0,
    batch_size: int = 128,
    shuffle_train: bool = True,
    shuffle_test: bool = False,
    random_state: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    as_loader: bool = True,
) -> Union[
    Tuple[DataLoader, DataLoader],
    Tuple[Dataset, Dataset]
]:
    """
    加载已知类数据，每个类别固定测试集样本数量

    载入文件夹内所有 .mat：
      1) 自动从文件名 'xxx_<label>.mat' 解析原始标签；
      2) 校验 known_class_count <= 可用 .mat 文件数；
      3) 选择"排序后前 K=known_class_count 个原始标签"并保留；
      4) 构造 RadarDataset（将保留的原始标签重映射为 0..K-1）；
      5) 按类别固定测试集数量进行分层切分。

    参数:
        test_per_class: 每个类别的测试集样本数量

    返回：(train, test)
    """
    root = Path(root)
    files = sorted(root.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"no .mat files under: {root}")

    # 提取目录中的"原始标签集合"
    raw_labels = []
    for f in files:
        lab = _extract_label_from_name(f)
        if lab is None:
            raise ValueError(
                f"file name does not match 'xxx_<label>.mat': {f.name}")
        raw_labels.append(lab)

    unique_labels = sorted(set(raw_labels))

    # 既满足"文件数"也满足"不同标签数"的语义
    if known_class_count > len(files) or known_class_count > len(unique_labels):
        raise ValueError(
            f"known_class_count={known_class_count} must be <= #files({len(files)}) "
            f"and <= #unique_labels({len(unique_labels)})."
        )
    if test_per_class <= 0:
        raise ValueError(f"test_per_class must be > 0, got {test_per_class}")

    # 选择前 K 个原始标签作为"已知类"
    kept_raw_labels = unique_labels[:known_class_count]

    # 构造数据集（仅保留所选标签，并重映射为 0..K-1）
    ds_full = RadarDataset(
        root,
        known_class_labels=kept_raw_labels,
        mat_key=mat_key,
        samples_axis=samples_axis,
    )

    # 固定测试集数量的分层切分
    tr_idx, te_idx = _stratified_split_fixed_test(
        labels=ds_full.labels.numpy(),
        test_per_class=test_per_class,
        random_state=random_state,
        shuffle=shuffle_train,
    )
    ds_train = Subset(ds_full, tr_idx)
    ds_test = Subset(ds_full, te_idx)

    if not as_loader:
        return ds_train, ds_test
    else:
        tr_loader = DataLoader(
            ds_train, batch_size=batch_size, shuffle=shuffle_train,
            num_workers=num_workers, pin_memory=pin_memory
        )
        te_loader = DataLoader(
            ds_test, batch_size=batch_size, shuffle=shuffle_test,
            num_workers=num_workers, pin_memory=pin_memory
        )
        return tr_loader, te_loader


def load_radar_unknown(
    root: Union[str, Path],
    known_class_count: int,
    *,
    mat_key: Optional[str] = "all_data",
    samples_axis: int = 0,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    as_loader: bool = True,
) -> Union[DataLoader, Dataset]:
    """
    从单一目录中加载“未知类测试集”：
      - 读取目录下所有 xxx_<label>.mat；
      - 按原始标签排序，前 K=known_class_count 视为“已知”（丢弃），其余为“未知”；
      - 未知样本统一打标签 -1。
    返回：DataLoader 或 Dataset（均满足 (Tensor, Tensor) 协议）。
    """
    root = Path(root)
    files = sorted(root.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"no .mat files under: {root}")

    # 解析所有原始标签
    raw_labels = []
    for f in files:
        lab = _extract_label_from_name(f)
        if lab is None:
            raise ValueError(
                f"file name does not match 'xxx_<label>.mat': {f.name}")
        raw_labels.append(lab)
    unique_labels = sorted(set(raw_labels))

    if known_class_count >= len(unique_labels):
        raise ValueError(
            f"known_class_count={known_class_count} leaves no unknown classes "
            f"(#unique_labels={len(unique_labels)})."
        )

    # 后半段 = 未知类
    unknown_raw_labels = unique_labels[known_class_count:]

    # 仅保留未知原始标签，读成基数据集（内部会重映射为 0..U-1）
    ds_unknown_base = RadarDataset(
        root,
        known_class_labels=unknown_raw_labels,
        mat_key=mat_key,
        samples_axis=samples_axis,
    )

    # 统一把标签覆写为 -1，保证 (Tensor, Tensor[long])
    ds_unknown = LabelOverride(ds_unknown_base, label=-1)

    if not as_loader:
        return ds_unknown

    return DataLoader(
        ds_unknown,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,   # 默认 collate 即可
    )
