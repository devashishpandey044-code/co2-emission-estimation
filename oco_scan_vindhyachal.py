import earthaccess, xarray as xr, numpy as np
import matplotlib.pyplot as plt
import os

# --- login + search (same 360) ---
earthaccess.login(persist=True)
results = earthaccess.search_data(
    short_name="OCO3_L2_Lite_FP", version="11r",
    temporal=("2020-01-01", "2020-12-31"),
    bounding_box=(68.0, 8.0, 89.0, 30.0),
)
print(f"Scanning {len(results)} granules for Vindhyachal coverage...\n")

PLANT_LAT, PLANT_LON = 24.10, 82.67
BOX = 0.5   # ~50 km box around the plant (tighter now)

os.makedirs("data/oco3_scan", exist_ok=True)

hits = []          # store (date, lats, lons, xco2) for days that cover the plant
kept_lat, kept_lon, kept_xco2 = [], [], []

for i, g in enumerate(results):
    try:
        # download one file at a time into a temp scan folder
        f = earthaccess.download([g], local_path="data/oco3_scan")[0]
        ds = xr.open_dataset(f)
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        xco2 = ds["xco2"].values
        qf  = ds["xco2_quality_flag"].values
        ds.close()

        good = (qf == 0) & np.isfinite(xco2)
        m = good & (lat > PLANT_LAT-BOX) & (lat < PLANT_LAT+BOX) & \
                   (lon > PLANT_LON-BOX) & (lon < PLANT_LON+BOX)
        n = int(m.sum())

        if n > 0:
            date = str(g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])[:10]
            print(f"  [{i+1}/{len(results)}]  {date}:  {n} soundings over Vindhyachal  <-- HIT")
            hits.append(date)
            kept_lat.extend(lat[m]); kept_lon.extend(lon[m]); kept_xco2.extend(xco2[m])
        else:
            if (i+1) % 30 == 0:
                print(f"  [{i+1}/{len(results)}] scanned, no hit yet...")

        os.remove(f)   # delete file after checking, to save disk
    except Exception as e:
        print(f"  [{i+1}] skipped ({str(e)[:50]})")

print(f"\n=== DONE ===")
print(f"Days with Vindhyachal coverage: {len(hits)}")
print(f"Total soundings collected over the plant: {len(kept_xco2)}")

if kept_xco2:
    kx = np.array(kept_xco2)
    print(f"XCO2 over Vindhyachal (all 2020): min={kx.min():.1f} max={kx.max():.1f} mean={kx.mean():.1f} ppm")
    # save the collected soundings so we never rescan
    np.savez("data/vindhyachal_soundings.npz",
             lat=np.array(kept_lat), lon=np.array(kept_lon), xco2=kx, dates=hits)
    print("Saved data/vindhyachal_soundings.npz")

    plt.figure(figsize=(8,7))
    sc = plt.scatter(kept_lon, kept_lat, c=kx, cmap="inferno", s=45)
    plt.colorbar(sc, label="XCO2 (ppm)")
    plt.scatter([PLANT_LON],[PLANT_LAT], marker="*", s=400,
                edgecolor="cyan", facecolor="none", linewidth=2, label="Vindhyachal")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title(f"OCO-3 XCO2 over Vindhyachal, 2020 ({len(hits)} days)")
    plt.legend(); plt.tight_layout()
    plt.savefig("data/vindhyachal_co2_year.png", dpi=130)
    print("Saved data/vindhyachal_co2_year.png")