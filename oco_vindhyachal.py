import earthaccess, xarray as xr, numpy as np
import matplotlib.pyplot as plt
import glob, os

# --- Vindhyachal coordinates (your Week-1 plant) ---
PLANT_LAT, PLANT_LON = 24.10, 82.67
BOX = 1.5   # degrees around the plant (~150 km each way) to start wide

# --- open the file we already downloaded ---
f = glob.glob("data/oco3/*.nc4")[0]
print("Reading:", f)
ds = xr.open_dataset(f)

lat = ds["latitude"].values
lon = ds["longitude"].values
xco2 = ds["xco2"].values
qf  = ds["xco2_quality_flag"].values

# keep only good-quality soundings (flag == 0 means best)
good = (qf == 0) & np.isfinite(xco2)
lat, lon, xco2 = lat[good], lon[good], xco2[good]
print(f"Good-quality soundings in file: {len(xco2)}")

# --- filter to the box around Vindhyachal ---
m = (lat > PLANT_LAT-BOX) & (lat < PLANT_LAT+BOX) & \
    (lon > PLANT_LON-BOX) & (lon < PLANT_LON+BOX)
plat, plon, pxco2 = lat[m], lon[m], xco2[m]
print(f"Soundings within {BOX} deg of Vindhyachal: {len(pxco2)}")

if len(pxco2) == 0:
    print("\nNo soundings over Vindhyachal in THIS file (expected — narrow track).")
    print("Day 5 will scan all 360 granules to find days that DO cover it.")
else:
    print(f"XCO2 near Vindhyachal: min={pxco2.min():.1f} max={pxco2.max():.1f} mean={pxco2.mean():.1f} ppm")
    # --- plot ---
    plt.figure(figsize=(8,7))
    sc = plt.scatter(plon, plat, c=pxco2, cmap="inferno", s=40)
    plt.colorbar(sc, label="XCO2 (ppm)")
    plt.scatter([PLANT_LON],[PLANT_LAT], marker="*", s=400,
                edgecolor="cyan", facecolor="none", linewidth=2, label="Vindhyachal plant")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("OCO-3 XCO2 near Vindhyachal power plant")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/vindhyachal_co2.png", dpi=130)
    print("\nSaved data/vindhyachal_co2.png")