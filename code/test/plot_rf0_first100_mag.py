import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


# Default paths (can be edited if needed)
MAT = os.path.join("data", "rf0", "rf0_0.mat")
OUT_DIR = os.path.join("data", "rf0")

# Top-level container and inner field per your rf0 schema:
#   data: 1x5000 struct/cell array
#   data(i).data: 2049x1 complex vector
TOP_VAR = "data"
INNER_FIELD = "data"


def to_list_1d(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, np.ndarray):
        return list(obj.ravel())
    raise TypeError("Top-level 'data' must be list/tuple/ndarray")


def get_field(sample, field):
    if isinstance(sample, dict) and field in sample:
        return sample[field]
    if hasattr(sample, field):
        return getattr(sample, field)
    if isinstance(sample, np.void) and sample.dtype.names and field in sample.dtype.names:
        return sample[field]
    raise KeyError(f"Sample missing field '{field}'")


def squeeze_1d(a):
    x = np.asarray(a).squeeze()
    if x.ndim == 2 and 1 in x.shape:
        x = x.reshape(-1)
    if x.ndim != 1:
        raise ValueError(f"Signal not 1D after squeeze: shape={x.shape}")
    return x


def load_first_n_magnitude(mat_path, top_var, inner_field, n=100):
    md = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    if top_var not in md:
        raise KeyError(f"'{top_var}' not found. keys={list(md.keys())}")
    container = md[top_var]
    items = to_list_1d(container)
    if len(items) < n:
        raise ValueError(f"Need at least {n} samples, got {len(items)}")
    sigs = []
    for i in range(n):
        raw = get_field(items[i], inner_field)
        x = squeeze_1d(raw)
        y = np.abs(x)  # magnitude
        sigs.append(y)
    L = min(len(s) for s in sigs)
    return np.vstack([s[:L] for s in sigs])  # (n, L)


def plot_batches(lines: np.ndarray, out_dir: str, base_name: str, batch=10):
    n, L = lines.shape
    os.makedirs(out_dir, exist_ok=True)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        sub = lines[start:end]
        rows = sub.shape[0]
        fig, axes = plt.subplots(nrows=rows, ncols=1, figsize=(12, 1.6 * rows), dpi=150, sharex=True)
        x = np.arange(L)
        if rows == 1:
            axes = [axes]
        for i in range(rows):
            ax = axes[i]
            ax.plot(x, sub[i], lw=0.9)
            ax.set_ylabel(f"#{start + i}")
            ax.grid(True, ls=":", alpha=0.4)
        axes[-1].set_xlabel("Index")
        title = f"{base_name} | magnitude | samples {start}-{end-1}"
        fig.suptitle(title)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{base_name}_mag_{start+1:02d}-{end:02d}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path)


def main():
    lines = load_first_n_magnitude(MAT, TOP_VAR, INNER_FIELD, n=100)
    base = os.path.splitext(os.path.basename(MAT))[0]
    plot_batches(lines, OUT_DIR, base, batch=10)


if __name__ == "__main__":
    main()

