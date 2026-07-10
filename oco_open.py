import earthaccess
import xarray as xr
import numpy as np
import os

# --- 1. Log in (uses saved credentials) ---
earthaccess.login(persist=True)

# --- 2. Same search that gave 360 ---
results = earthaccess.search_data(
    short_name="OCO3_L2_Lite_FP", version="11r",
    temporal=("2020-01-01", "2020-12-31"),
    bounding_box=(68.0, 8.0, 89.0, 30.0),
)
print(f"Found {len(results)} granules.")

# --- 3. Download just the FIRST one ---
os.makedirs("data/oco3", exist_ok=True)
print("\nDownloading granule 0 ...")
files = earthaccess.download(results[0:1], local_path="data/oco3")
print("Downloaded:", files)

# --- 4. Open and inspect ---
ds = xr.open_dataset(files[0])
print("\n=== What's in this file ===")
print("Number of soundings:", ds.sizes.get("sounding_id", "unknown"))

print("\nKey variables present:")
for v in ["xco2", "latitude", "longitude", "xco2_quality_flag"]:
    print("  ", v, "->", "YES" if v in ds.variables else "NO")

if "xco2" in ds.variables:
    x = ds["xco2"].values
    x = x[np.isfinite(x)]
    print(f"\nXCO2: min={x.min():.1f}  max={x.max():.1f}  mean={x.mean():.1f} ppm")
    print(f"Total soundings: {len(x)}")