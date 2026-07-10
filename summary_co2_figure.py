import json, numpy as np
import matplotlib.pyplot as plt

# --- load your real results ---
res = json.load(open("data/plant_results.json"))
# order plants consistently
order = ["Tirora", "Vindhyachal", "Sasan", "Mundra"]
res = sorted(res, key=lambda r: order.index(r["plant"]) if r["plant"] in order else 99)

names   = [r["plant"] for r in res]
enh     = [r["co2_enhancement_ppm"] for r in res]
bgstd   = [r["bg_std_ppm"] for r in res]
no2km   = [r["no2_peak_km"] for r in res]
sound   = [r["soundings"] for r in res]
winddiff= [r["wind_co2_diff_deg"] for r in res]

# handle Mundra's null enhancement gracefully
enh_plot = [e if e is not None else 0 for e in enh]

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Satellite CO$_2$ Characterization of Indian Coal Power Plants (OCO-3, 2020)",
             fontsize=15, fontweight="bold")

# ---------- Panel 1: CO2 enhancement vs background scatter ----------
ax = axs[0,0]
x = np.arange(len(names))
bars = ax.bar(x, enh_plot, color=["#27ae60" if (e is not None and e>s) else "#e67e22" if e is not None else "#aaaaaa"
                                  for e,s in zip(enh,bgstd)])
# background-scatter error bars (the honesty layer)
for i,(e,s) in enumerate(zip(enh,bgstd)):
    if e is not None:
        ax.errorbar(i, e, yerr=s, fmt="none", ecolor="black", capsize=6, lw=1.5)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("CO$_2$ enhancement (ppm)")
ax.set_title("CO$_2$ enhancement vs background scatter\n(green = signal exceeds noise)")
for i,e in enumerate(enh):
    label = f"{e:+.2f}" if e is not None else "no data\n(low coverage)"
    ax.text(i, (enh_plot[i]+0.05 if e is not None else 0.1), label, ha="center", fontweight="bold", fontsize=9)
ax.axhline(0, color="gray", lw=0.8)

# ---------- Panel 2: NO2 co-location distance ----------
ax = axs[0,1]
colors = ["#27ae60" if d<=10 else "#e67e22" if d<=30 else "#c0392b" for d in no2km]
ax.bar(x, no2km, color=colors)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("NO$_2$ peak distance from plant (km)")
ax.set_title("NO$_2$ co-location\n(green <10 km = bulls-eye)")
ax.axhline(10, ls="--", color="gray", label="10 km")
for i,d in enumerate(no2km):
    ax.text(i, d+1, f"{d:.0f} km", ha="center", fontweight="bold", fontsize=9)
ax.legend()

# ---------- Panel 3: coverage (soundings) ----------
ax = axs[1,0]
ax.bar(x, sound, color="#2c3e50")
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("OCO-3 soundings collected")
ax.set_title("Data coverage per plant\n(more = more reliable)")
ax.axhline(20, ls="--", color="red", label="min. threshold (20)")
for i,s in enumerate(sound):
    ax.text(i, s+max(sound)*0.01, str(s), ha="center", fontweight="bold", fontsize=9)
ax.legend()

# ---------- Panel 4: summary verdict table ----------
ax = axs[1,1]; ax.axis("off")
ax.set_title("Overall assessment", fontweight="bold")
def verdict(r):
    if r["co2_enhancement_ppm"] is None: return "Insufficient coverage"
    strong = (r["co2_enhancement_ppm"] > r["bg_std_ppm"])
    close  = (r["no2_peak_km"] <= 10)
    wind   = (r["wind_co2_diff_deg"] is not None and r["wind_co2_diff_deg"] < 45)
    score = sum([strong, close, wind])
    return {3:"Strong (all 3)",2:"Good (2/3)",1:"Moderate (1/3)",0:"Weak"}[score]
rows = [[r["plant"],
        f'{r["co2_enhancement_ppm"]:+.2f}' if r["co2_enhancement_ppm"] is not None else "—",
        f'{r["no2_peak_km"]:.0f} km',
        f'{r["soundings"]}',
        verdict(r)] for r in res]
tbl = ax.table(cellText=rows,
    colLabels=["Plant","CO2 enh.","NO2 dist.","Soundings","Verdict"],
    cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
for j in range(5):
    tbl[0,j].set_facecolor("#0b3d5c"); tbl[0,j].set_text_props(color="white", fontweight="bold")

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("data/summary_co2_plants.png", dpi=140, bbox_inches="tight")
print("Saved data/summary_co2_plants.png")