import numpy as np, glob, os
import matplotlib.pyplot as plt

def load_one(folder, contains=None):
    files = sorted(glob.glob(f"{folder}/*.npy"))
    if contains:
        files = [f for f in files if contains.lower() in os.path.basename(f).lower()]
    if not files:
        return None, None
    f = files[len(files)//2]                 # pick a middle month
    arr = np.load(f).astype(np.float32)
    return arr, os.path.basename(f).replace(".npy", "")

# pick one representative tile of each type
samples = [
    ("PLANT (point source)",   load_one("data/monthly/positive", "VINDH")),
    ("CITY (diffuse)",         load_one("data/monthly/hard_negative", "city_Delhi")),
    ("INDUSTRY",               load_one("data/monthly/hard_negative", "ind_Jamshedpur")),
    ("HIGHWAY (linear)",       load_one("data/monthly/hard_negative", "hwy_")),
    ("RURAL (empty)",          load_one("data/monthly/negative", "rural")),
]

samples = [(t, a, n) for t, (a_n) in
           [(t, ld) for t, ld in samples] for a, n in [a_n]]

fig, axes = plt.subplots(1, len(samples), figsize=(4*len(samples), 4.5))

# use a shared colour scale so tiles are comparable
allvals = np.concatenate([a[np.isfinite(a)].ravel() for _, a, _ in samples if a is not None])
vmax = np.nanpercentile(allvals, 98)

for ax, (title, arr, name) in zip(axes, samples):
    if arr is None:
        ax.set_title(f"{title}\n(not found)", fontsize=9); ax.axis("off"); continue
    m = np.nanmean(arr)
    arr_f = np.where(np.isfinite(arr), arr, np.nanmean(arr))
    im = ax.imshow(arr_f, cmap="jet", vmin=0, vmax=vmax)
    ax.set_title(f"{title}\n{name}\nmean={m:.2e}", fontsize=8)
    ax.axis("off")

fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
plt.savefig("data/comparison_types.png", dpi=120, bbox_inches="tight")
print("Saved data/comparison_types.png")

# also print the mean NO2 of each type so you can compare numbers
print("\nMean NO2 by type (all tiles):")
for label, folder in [("plant", "positive"), ("hard_neg", "hard_negative"), ("rural", "negative")]:
    vals = [np.nanmean(np.load(f)) for f in glob.glob(f"data/monthly/{folder}/*.npy")]
    print(f"  {label:10s} n={len(vals):3d}  mean={np.mean(vals):.3e}  max={np.max(vals):.3e}")