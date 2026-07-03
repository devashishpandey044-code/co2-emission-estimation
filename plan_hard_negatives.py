import pandas as pd
import numpy as np

# ---------- your candidate hard negatives ----------
# These have HIGH NO2 but are NOT power plants — the whole point.
hard_negatives = [
    # --- Big cities (traffic + density NO2) ---
    ("city_Delhi",        28.61, 77.21),
    ("city_Mumbai",       19.08, 72.88),
    ("city_Kolkata",      22.57, 88.36),
    ("city_Bangalore",    12.97, 77.59),
    ("city_Hyderabad",    17.39, 78.49),
    # --- Industrial / non-plant heavy industry ---
    ("ind_Jamshedpur",    22.80, 86.18),   # steel city
    ("ind_Bhilai",        21.19, 81.38),   # steel city
    ("ind_Kanpur",        26.45, 80.33),   # heavy industry / tanneries
    ("ind_Ludhiana",      30.90, 75.85),   # industrial hub
    ("ind_Surat",         21.17, 72.83),   # industrial / textiles
    # --- Highway corridors (linear NO2 from trucks) ---
    ("hwy_DelhiJaipur",   27.60, 76.60),   # NH48 corridor, open stretch
    ("hwy_MumbaiPune",    18.75, 73.50),   # expressway stretch
    ("hwy_AgraKanpur",    27.00, 79.20),   # NH19 stretch
    ("hwy_AhmedabadVadodara", 22.70, 72.90),   # NH53 stretch
    ("hwy_GTRoad_UP",     26.90, 81.90),   # Grand Trunk Road stretch
]

# ---------- load your plants ----------
plants = pd.read_csv("data/top5_plants.csv")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ---------- sanity check: distance to nearest plant ----------
SAFE_KM = 80   # a hard negative should be at least this far from any plant

print(f"{'name':22s} {'nearest plant':22s} {'dist_km':>8s}  status")
print("-" * 70)
rows = []
for name, lat, lon in hard_negatives:
    dmin, nearest = 1e9, ""
    for _, p in plants.iterrows():
        d = haversine_km(lat, lon, p["latitude"], p["longitude"])
        if d < dmin:
            dmin, nearest = d, str(p["name"])
    status = "OK" if dmin >= SAFE_KM else "!! TOO CLOSE"
    print(f"{name:22s} {nearest[:22]:22s} {dmin:8.0f}  {status}")
    rows.append((name, lat, lon, round(dmin, 1), status))

# ---------- save the clean list ----------
df = pd.DataFrame(rows, columns=["name", "latitude", "longitude", "dist_to_plant_km", "status"])
clean = df[df["status"] == "OK"][["name", "latitude", "longitude"]]
clean.to_csv("data/hard_negatives.csv", index=False)

n_ok = (df["status"] == "OK").sum()
n_bad = (df["status"] != "OK").sum()
print("-" * 70)
print(f"OK: {n_ok}   TOO CLOSE: {n_bad}")
print("Saved clean list -> data/hard_negatives.csv")
if n_bad:
    print("\nReview the TOO CLOSE points above — I'll help you replace them.")