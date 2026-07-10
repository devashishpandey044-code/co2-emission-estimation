import numpy as np, glob, os

def pair_folder(no2_dir, so2_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    no2_files = {os.path.basename(f): f for f in glob.glob(f"{no2_dir}/*.npy")}
    so2_files = {os.path.basename(f): f for f in glob.glob(f"{so2_dir}/*.npy")}
    common = sorted(set(no2_files) & set(so2_files))
    saved = 0
    for name in common:
        no2 = np.load(no2_files[name]).astype(np.float32)
        so2 = np.load(so2_files[name]).astype(np.float32)
        if no2.shape != so2.shape:
            continue
        # fill gaps with each channel's own mean
        no2 = np.where(np.isfinite(no2), no2, np.nanmean(no2))
        so2 = np.where(np.isfinite(so2), so2, np.nanmean(so2))
        # SO2 noise can go negative — clamp at 0 (can't have negative concentration)
        so2 = np.clip(so2, 0, None)
        stacked = np.stack([no2, so2], axis=0)   # shape (2, 64, 64)
        np.save(f"{out_dir}/{name}", stacked)
        saved += 1
    print(f"{out_dir:32s} paired={saved}  (NO2={len(no2_files)}, SO2={len(so2_files)})")
    return saved

os.makedirs("data/twoch", exist_ok=True)
p = pair_folder("data/monthly/positive",      "data/so2/positive",      "data/twoch/positive")
h = pair_folder("data/monthly/hard_negative", "data/so2/hard_negative", "data/twoch/hard_negative")
r = pair_folder("data/monthly/negative",      "data/so2/negative",      "data/twoch/negative")
print(f"\nTotal 2-channel tiles: plants={p}  hard_neg={h}  rural={r}")
print("Each tile shape = (2, 64, 64): channel 0 = NO2, channel 1 = SO2")