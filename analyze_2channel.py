import numpy as np, glob, os, torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Detector2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2,16,3,padding=1), nn.BatchNorm2d(16), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64,2))
    def forward(self,x): return self.net(x)

model = Detector2().to(DEVICE)
model.load_state_dict(torch.load("detector2_2ch_hard_only.pt",
                                 map_location=DEVICE, weights_only=True))
model.eval()

# normalization reference: plants + hard (same as training)
P = [np.load(f) for f in glob.glob("data/twoch/positive/*.npy")]
H = [np.load(f) for f in glob.glob("data/twoch/hard_negative/*.npy")]
allX = np.stack(P+H)
chmean = [allX[:,c].mean() for c in range(2)]
chstd  = [allX[:,c].std()+1e-12 for c in range(2)]

def predict(arr):
    x = arr.copy().astype(np.float32)
    for c in range(2):
        x[c] = (x[c]-chmean[c])/chstd[c]
    t = torch.tensor(x[None], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return torch.softmax(model(t),1)[0,1].item()

# score every hard negative by how often it's called "plant"
scores = {}
for f in glob.glob("data/twoch/hard_negative/*.npy"):
    base = os.path.basename(f).rsplit("_",2)[0]
    scores.setdefault(base,[]).append(predict(np.load(f)))

print("Hard negatives most mistaken for PLANTS (2-channel NO2+SO2 model):")
print(f"{'location':22s} {'avg P(plant)':>12s}")
for base,ps in sorted(scores.items(), key=lambda kv:-np.mean(kv[1])):
    tag = "  <- STEEL" if base.startswith("ind_") else ("  <- city" if base.startswith("city_") else "")
    print(f"  {base:22s} {np.mean(ps):>10.2f}{tag}")