import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Paths
MAT = os.path.join("data", "rf0", "rf0_0.mat")
OUT_DIR = os.path.join("data", "rf0")

# Top-level variable name holding 1x5000 samples
TOP_VAR = "data"  # data is 1x5000, each element has field 'data' which is 2049x1
# Field name inside each sample that contains the 2049x1 complex vector
FIELD = "data"


def to_list_1d(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, np.ndarray):
        # Flatten cell/object/struct array to 1D list
        return list(obj.ravel())
    raise TypeError("Expected array-like for top-level samples container")


def get_field(sample, field_name):
    # Supports dict, mat_struct, structured arrays
    if isinstance(sample, dict) and field_name in sample:
        return sample[field_name]
    if hasattr(sample, field_name):
        return getattr(sample, field_name)
    if isinstance(sample, np.void) and sample.dtype.names and field_name in sample.dtype.names:
        return sample[field_name]
    raise KeyError(f"Sample missing field '{field_name}'")


def squeeze_signal(a):
    x = np.asarray(a).squeeze()
    if x.ndim == 2 and 1 in x.shape:
        x = x.reshape(-1)
    if x.ndim != 1:
        raise ValueError(f"Signal not 1D after squeeze: shape={x.shape}")
    return x


def load_first_n_signals(mat_path, top_var, field, n=10):
    md = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    if top_var not in md:
        raise KeyError(f"'{top_var}' not found. keys={list(md.keys())}")
    container = md[top_var]
    items = to_list_1d(container)
    if len(items) < n:
        raise ValueError(f"Need at least {n} samples, got {len(items)}")
    sigs = []
    for i in range(n):
        raw = get_field(items[i], field)
        sig = squeeze_signal(raw)
        sigs.append(sig)
    # Align lengths just in case
    L = min(len(s) for s in sigs)
    return np.vstack([s[:L] for s in sigs])  # (n, L)


def plot_rows(data2d, title, out_path):
    n, L = data2d.shape
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 1.6 * n), dpi=150, sharex=True)
    x = np.arange(L)
    for i in range(n):
        ax = axes[i] if isinstance(axes, np.ndarray) else axes
        ax.plot(x, data2d[i], lw=0.9)
        ax.set_ylabel(f"#{i}")
        ax.grid(True, ls=":", alpha=0.4)
    if isinstance(axes, np.ndarray):
        axes[-1].set_xlabel("Index")
    else:
        axes.set_xlabel("Index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    sigs = load_first_n_signals(MAT, TOP_VAR, FIELD, n=10)  # complex allowed
    # Prepare three views
    mag = np.abs(sigs)
    real = np.real(sigs)
    imag = np.imag(sigs)

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(MAT))[0]
    out_mag = os.path.join(OUT_DIR, f"{base}_first10_mag.png")
    out_real = os.path.join(OUT_DIR, f"{base}_first10_real.png")
    out_imag = os.path.join(OUT_DIR, f"{base}_first10_imag.png")

    plot_rows(mag, f"{base} | Amplitude (linear)", out_mag)
    plot_rows(real, f"{base} | Real part", out_real)
    plot_rows(imag, f"{base} | Imag part", out_imag)

    print("Saved:")
    print(" ", out_mag)
    print(" ", out_real)
    print(" ", out_imag)


if __name__ == "__main__":
    main()

