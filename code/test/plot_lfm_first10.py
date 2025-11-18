import os
import numpy as np
import matplotlib.pyplot as plt


MAT = "data1\data_noise_30\data7.mat"
OUT = "data1\data_noise_30\data_7_first10.png"


def _try_import_scipy():
    try:
        import scipy.io as sio  # type: ignore
        return sio
    except Exception:
        return None


def _try_import_h5py():
    try:
        import h5py  # type: ignore
        return h5py
    except Exception:
        return None


def _is_numeric_array(x) -> bool:
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)


def _choose_array_from_dict(d: dict):
    d = {k: v for k, v in d.items() if not k.startswith("__")}
    # Heuristic: prefer common names
    for key in ("data", "X", "samples", "S", "Y"):
        if key in d and _is_numeric_array(d[key]):
            return key, np.asarray(d[key])
    # Fallback: largest numeric array
    best_key, best_arr, best_size = None, None, -1
    for k, v in d.items():
        if _is_numeric_array(v):
            arr = np.asarray(v)
            sz = int(np.prod(arr.shape))
            if sz > best_size:
                best_key, best_arr, best_size = k, arr, sz
    if best_arr is None:
        raise RuntimeError("No numeric ndarray found in MAT file.")
    return best_key, best_arr


def load_primary_array(mat_path: str):
    # Try SciPy (MAT v7 and earlier)
    sio = _try_import_scipy()
    if sio is not None:
        try:
            md = sio.loadmat(mat_path, squeeze_me=True, simplify_cells=True)
            key, arr = _choose_array_from_dict(md)
            return key, np.asarray(arr)
        except Exception:
            pass

    # Try h5py (MAT v7.3)
    h5py = _try_import_h5py()
    if h5py is not None:
        try:
            with h5py.File(mat_path, "r") as f:
                best_path, best_obj, best_size = None, None, -1
                def visit(name, obj):
                    nonlocal best_path, best_obj, best_size
                    if isinstance(obj, h5py.Dataset):
                        dt = obj.dtype
                        if np.issubdtype(dt, np.number):
                            size = int(np.prod(obj.shape)) if obj.shape else 0
                            if size > best_size:
                                best_path, best_obj, best_size = name, obj, size
                f.visititems(visit)
                if best_obj is None:
                    raise RuntimeError("No numeric datasets found via h5py.")
                data = best_obj[...]
                return best_path, np.asarray(data)
        except Exception:
            pass

    raise RuntimeError("Failed to load MAT file with scipy or h5py.")


def first_10_rows(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    # If 1D, cannot take 10 rows. In that case, split into 10 equal parts.
    if a.ndim == 1:
        L = a.shape[0] // 10
        if L <= 0:
            raise ValueError("1D array too short for 10 samples.")
        return np.vstack([a[i*L:(i+1)*L] for i in range(10)])
    # If 2D or higher, pick a sample axis with size >= 10 and make rows
    axes = [i for i, s in enumerate(a.shape) if s >= 10]
    if not axes:
        raise ValueError("No axis with >=10 to serve as samples.")
    ax = axes[0]
    moved = np.moveaxis(a, ax, 0)  # (S, ...)
    rest = int(np.prod(moved.shape[1:])) or 1
    flat = moved.reshape(moved.shape[0], rest)
    return flat[:10, :]


def plot_rows(lines: np.ndarray, out_path: str, title: str):
    n, L = lines.shape
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols = 5, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows), dpi=150, sharex=True)
    axes = np.asarray(axes)
    x = np.arange(L)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        if idx < n:
            ax.plot(x, lines[idx], lw=0.9)
            ax.set_title(f"Sample {idx + 1}", fontsize=9)
            ax.grid(True, ls=":", alpha=0.4)
        else:
            ax.axis("off")

        if r == rows - 1:
            ax.set_xlabel("Index")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def main():
    key, arr = load_primary_array(MAT)
    # Real-only dataset expected; if complex sneaks in, take real part
    arr = np.real(arr)
    lines = first_10_rows(arr)
    title = f"{os.path.basename(MAT)} | var: {key} | first 10 samples"
    plot_rows(lines, OUT, title)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()

