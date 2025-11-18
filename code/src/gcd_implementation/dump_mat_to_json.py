#!/usr/bin/env python
"""
将目录中的 .mat 文件批量转换为 JSON，便于快速查看标签与数据。

默认读取键 `all_data`，可通过 --mat-key 修改；若键缺失则会自动寻找首个 ndarray。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from scipy.io import loadmat
except ImportError as exc:  # pragma: no cover - 依赖缺失时直接提示
    raise SystemExit(
        "缺少 SciPy，无法读取 .mat 文件。请先安装依赖：pip install scipy"
    ) from exc


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_code_mod = sys.modules.get("code")
if _code_mod is not None and not hasattr(_code_mod, "__path__"):
    sys.modules.pop("code")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量将 .mat 文件导出为 JSON（包含元数据与数组内容）"
    )
    parser.add_argument(
        "mat_root",
        type=str,
        help="包含 .mat 文件的根目录",
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="JSON 输出目录，将按原始目录结构创建子目录",
    )
    parser.add_argument(
        "--mat-key",
        type=str,
        default="all_data",
        help="读取 .mat 时使用的键；若留空则自动选择首个 ndarray",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="仅转换前 N 个 .mat 文件，调试时可使用",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="将数组转换为 float32，以避免浮点精度过高导致文件膨胀",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="使用缩进格式化 JSON（体积会更大，仅建议调试时启用）",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="遇到单个文件出错时继续处理后续文件",
    )
    return parser.parse_args()


def _resolve_array(mat_path: Path, raw: Dict[str, Any], mat_key: str | None) -> np.ndarray:
    if mat_key:
        if mat_key not in raw:
            usable = [k for k in raw if not k.startswith("__")]
            raise KeyError(f"{mat_path.name}: key '{mat_key}' 不存在，可选键 {usable}")
        arr = raw[mat_key]
    else:
        arr = next(
            (
                v
                for k, v in raw.items()
                if not k.startswith("__") and isinstance(v, np.ndarray)
            ),
            None,
        )
        if arr is None:
            raise KeyError(f"{mat_path.name}: 未找到 ndarray 数据")
    return np.asarray(arr)


def _to_json_serializable(array: np.ndarray, cast_float32: bool) -> Dict[str, Any]:
    if cast_float32 and np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)

    payload: Dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "data": array.tolist(),
    }
    return payload


def dump_mat_files(args: argparse.Namespace) -> None:
    mat_root = Path(args.mat_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not mat_root.is_dir():
        raise SystemExit(f"输入目录不存在: {mat_root}")

    mat_files: List[Path] = sorted(mat_root.rglob("*.mat"))
    if not mat_files:
        raise SystemExit(f"在 {mat_root} 下未找到任何 .mat 文件")

    if args.max_files is not None:
        mat_files = mat_files[: max(args.max_files, 0)]

    output_root.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None

    for mat_path in mat_files:
        rel_path = mat_path.relative_to(mat_root)
        out_path = (output_root / rel_path).with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            raw = loadmat(mat_path.as_posix())
            array = _resolve_array(mat_path, raw, args.mat_key or None)
            payload = {
                "source": str(rel_path),
                "mat_key": args.mat_key or "auto",
                **_to_json_serializable(array, args.float32),
            }
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=indent)
        except Exception as exc:
            if args.skip_errors:
                print(f"[WARN] {mat_path}: {exc}")
                continue
            raise

        print(f"写出 {out_path}")


def main() -> None:
    args = parse_args()
    dump_mat_files(args)


if __name__ == "__main__":
    main()
