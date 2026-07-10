import numpy as np, glob, os

def group_stats(no2_dir, so2_dir, label):
    no2_files = {os.path.basename(f): f for f in glob.glob(f"{no2_dir}/*.npy")}
    so2_files = {os.path.basename(f): f for f in glob.glob(f"{so2_dir}/*.npy")}
    common = sorted(set(no2_files) & set(so2_files))
    no2_means, so2_means, ratios = [], [], []
    for name in common:
        no2 = np.load(no2_files[name]); so2 = np.load(so2_files[name])
        n = np.nanmean(no2)
        s = np.nanmean(so2)
        # clamp tiny/negative SO2 noise to 0 for the ratio
        s_clip = max(s, 0.0)
        no2_means.append(n); so2_means.append(s)
        if n > 0:
            ratios.append(s_clip / n)
    print(f"{label:14s} n={len(common):3d}  "
          f"NO2={np.mean(no2_means):.3e}  "
          f"SO2={np.mean(so2_means):.3e}  "
          f"SO2/NO2={np.mean(ratios):.3f}")
    return np.mean(no2_means), np.mean(so2_means), np.mean(ratios)

print(f"{'group':14s} {'':3s}  {'meanNO2':>10s}  {'meanSO2':>10s}  {'ratio':>8s}")
print("-"*60)
group_stats("data/monthly/positive",      "data/so2/positive",      "PLANTS")
group_stats("data/monthly/hard_negative", "data/so2/hard_negative", "HARD_NEG(all)")
group_stats("data/monthly/negative",      "data/so2/negative",      "RURAL")

# break hard negatives into sub-types
print("\n--- hard negatives by type ---")
for prefix, lab in [("city_", "CITIES"), ("ind_", "INDUSTRY"), ("hwy_", "HIGHWAYS")]:
    no2f = {os.path.basename(f): f for f in glob.glob("data/monthly/hard_negative/*.npy") if os.path.basename(f).startswith(prefix)}
    so2f = {os.path.basename(f): f for f in glob.glob("data/so2/hard_negative/*.npy") if os.path.basename(f).startswith(prefix)}
    common = sorted(set(no2f) & set(so2f))
    if not common:
        print(f"{lab}: none"); continue
    ns, ss, rs = [], [], []
    for name in common:
        n = np.nanmean(np.load(no2f[name])); s = np.nanmean(np.load(so2f[name]))
        ns.append(n); ss.append(s)
        if n>0: rs.append(max(s,0)/n)
    print(f"{lab:10s} n={len(common):3d}  NO2={np.mean(ns):.3e}  SO2={np.mean(ss):.3e}  SO2/NO2={np.mean(rs):.3f}")

print("\nKEY QUESTION: do PLANTS have a higher SO2/NO2 ratio than CITIES/INDUSTRY?")