import numpy as np
import matplotlib.pyplot as plt

# --- load the saved soundings (no re-downloading) ---
d = np.load("data/vindhyachal_soundings.npz")
lat, lon, xco2 = d["lat"], d["lon"], d["xco2"]
print(f"Loaded {len(xco2)} soundings over Vindhyachal.")

PLANT_LAT, PLANT_LON = 24.10, 82.67

# distance of each sounding from the plant (in degrees, ~111 km/deg)
dist = np.sqrt((lat-PLANT_LAT)**2 + (lon-PLANT_LON)**2)

# --- define zones ---
NEAR = 0.25     # inner circle ~25 km = "near plant"
BG_IN, BG_OUT = 0.4, 0.9   # ring from ~45km to ~100km = "background"

near_mask = dist < NEAR
bg_mask   = (dist > BG_IN) & (dist < BG_OUT)

near_co2 = xco2[near_mask]
bg_co2   = xco2[bg_mask]

print(f"\nNear-plant soundings (<{NEAR*111:.0f} km): {len(near_co2)}")
print(f"Background soundings (ring):            {len(bg_co2)}")

if len(near_co2) > 5 and len(bg_co2) > 5:
    near_mean = near_co2.mean()
    bg_mean   = bg_co2.mean()
    enhancement = near_mean - bg_mean

    print(f"\n=== ENHANCEMENT TEST ===")
    print(f"  Mean XCO2 near plant : {near_mean:.2f} ppm")
    print(f"  Mean XCO2 background : {bg_mean:.2f} ppm")
    print(f"  ENHANCEMENT          : {enhancement:+.2f} ppm")
    print(f"  Background std       : {bg_co2.std():.2f} ppm")

    # is the enhancement bigger than the natural scatter?
    if enhancement > bg_co2.std():
        print(f"\n  -> Enhancement EXCEEDS background noise. Promising signal!")
    else:
        print(f"\n  -> Enhancement is within background scatter. Weak/uncertain.")

    # --- visualize the two distributions ---
    plt.figure(figsize=(9,5))
    plt.hist(bg_co2, bins=40, alpha=0.6, label=f"Background ({len(bg_co2)})", color="steelblue", density=True)
    plt.hist(near_co2, bins=25, alpha=0.6, label=f"Near plant ({len(near_co2)})", color="crimson", density=True)
    plt.axvline(bg_mean, color="steelblue", ls="--")
    plt.axvline(near_mean, color="crimson", ls="--")
    plt.xlabel("XCO2 (ppm)"); plt.ylabel("normalized count")
    plt.title(f"Vindhyachal: near-plant vs background CO2  (enhancement {enhancement:+.2f} ppm)")
    plt.legend(); plt.tight_layout()
    plt.savefig("data/vindhyachal_enhancement.png", dpi=130)
    print("\nSaved data/vindhyachal_enhancement.png")
else:
    print("\nNot enough soundings in one of the zones - we'll widen the zones.")