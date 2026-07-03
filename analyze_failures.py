import numpy as np, glob, os, torch
import torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64, 2))
    def forward(self, x): return self.net(x)

# load the hard_only model (the interesting one)
model = Detector().to(DEVICE)
model.load_state_dict(torch.load("detector_hard_only.pt", map_location=DEVICE))
model.eval()

# we need the SAME normalization as training: fit on plants+hard
def load_folder(path):
    items = []
    for f in sorted(glob.glob(f"{path}/*.npy")):
        arr = np.load(f).astype(np.float32)
        arr = np.where(np.isfinite(arr), arr, np.nanmean(arr))
        items.append((os.path.basename(f).replace(".npy",""), arr))
    return items

plants = load_folder("data/monthly/positive")
hards  = load_folder("data/monthly/hard_negative")

allX = np.stack([a for _,a in plants] + [a for _,a in hards])
mean, std = allX.mean(), allX.std()+1e-12

def predict(arr):
    x = ((arr - mean)/std)[None,None,:,:]
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32).to(DEVICE))
        prob = torch.softmax(logits,1)[0,1].item()   # P(plant)
    return prob

# find which HARD NEGATIVES get called "plant" (false alarms)
print("Hard negatives most often mistaken for PLANTS:")
scores = {}
for name, arr in hards:
    base = name.rsplit("_",2)[0]              # strip _YYYY_MM
    scores.setdefault(base, []).append(predict(arr))
ranked = sorted(scores.items(), key=lambda kv: -np.mean(kv[1]))
for base, ps in ranked:
    print(f"  {base:22s} avg P(plant)={np.mean(ps):.2f}  (n={len(ps)})")

# visualize the 4 worst false alarms
worst = []
for name, arr in hards:
    worst.append((predict(arr), name, arr))
worst.sort(key=lambda t: -t[0])
fig, axes = plt.subplots(1, 4, figsize=(16,4.5))
for ax,(p,name,arr) in zip(axes, worst[:4]):
    im = ax.imshow(arr, cmap="jet")
    ax.set_title(f"{name}\nP(plant)={p:.2f}  (WRONG)", fontsize=8)
    ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig("data/worst_false_alarms.png", dpi=120, bbox_inches="tight")
print("\nSaved data/worst_false_alarms.png")