import numpy as np
import matplotlib.pyplot as plt
import glob, os

files = sorted(glob.glob("data/tiles/*.npy"))

if not files:
    print("No tiles found in data/tiles/. Run export_tiles.py first.")
    raise SystemExit

n = len(files)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
if n == 1:
    axes = [axes]

for ax, f in zip(axes, files):
    arr = np.load(f)
    name = os.path.basename(f).replace(".npy", "")

    # report basic stats so you can spot empty/gappy tiles
    valid = np.isfinite(arr)
    pct_gap = 100 * (1 - valid.mean())
    mean_val = np.nanmean(arr)
    print(f"{name:30s} shape={arr.shape}  mean={mean_val:.3e}  gaps={pct_gap:4.1f}%")

    im = ax.imshow(arr, cmap="jet")
    ax.set_title(f"{name}\nmean={mean_val:.2e}", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("data/tiles_overview.png", dpi=120, bbox_inches="tight")
print("\nSaved data/tiles_overview.png")