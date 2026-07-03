import numpy as np, glob, os, torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.SiLU())
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64, 2))
    def forward(self, x):
        f = self.features(x)
        return self.head(f)

# NOTE: this splits the same architecture into features/head so we can hook
# the last conv layer. We load the trained weights by matching parameter order.
model = Detector().to(DEVICE)
state = torch.load("detector_hard_only.pt", map_location=DEVICE, weights_only=True)
# remap keys from the old sequential "net.*" naming into features/head
new_state = {}
old = list(state.values())
new_keys = list(model.state_dict().keys())
for k, v in zip(new_keys, old):
    new_state[k] = v
model.load_state_dict(new_state)
model.eval()

# ----- Grad-CAM hook on the last conv layer (index 8 in features) -----
activations, gradients = {}, {}
last_conv = model.features[8]   # the SiLU after the 3rd conv; use conv output
target_conv = model.features[6] # the 3rd Conv2d (64 filters)
def fwd_hook(m,i,o): activations["v"] = o.detach()
def bwd_hook(m,gi,go): gradients["v"] = go[0].detach()
target_conv.register_forward_hook(fwd_hook)
target_conv.register_full_backward_hook(bwd_hook)

def normalize_ref():
    P = [np.load(f) for f in glob.glob("data/monthly/positive/*.npy")]
    H = [np.load(f) for f in glob.glob("data/monthly/hard_negative/*.npy")]
    allX = np.stack([np.where(np.isfinite(a), a, np.nanmean(a)) for a in P+H])
    return allX.mean(), allX.std()+1e-12
MEAN, STD = normalize_ref()

def gradcam(arr):
    a = np.where(np.isfinite(arr), arr, np.nanmean(arr))
    x = torch.tensor(((a-MEAN)/STD)[None,None], dtype=torch.float32, device=DEVICE)
    x.requires_grad_(True)
    logits = model(x)
    cls = logits.argmax(1)
    model.zero_grad()
    logits[0, 1].backward()          # gradient wrt "plant" class
    g = gradients["v"][0]            # (C,h,w)
    A = activations["v"][0]         # (C,h,w)
    weights = g.mean(dim=(1,2))     # importance per channel
    cam = F.relu((weights[:,None,None]*A).sum(0))
    cam = cam / (cam.max()+1e-8)
    cam = F.interpolate(cam[None,None], size=arr.shape, mode="bilinear")[0,0]
    return cam.cpu().numpy(), torch.softmax(logits,1)[0,1].item()

# pick 4 misclassified hard negatives (industrial/city) to explain
targets = []
for pat in ["ind_Bhilai", "ind_Jamshedpur", "city_Delhi", "city_Mumbai"]:
    fs = sorted(glob.glob(f"data/monthly/hard_negative/{pat}_*.npy"))
    if fs: targets.append((pat, fs[len(fs)//2]))

fig, axes = plt.subplots(2, len(targets), figsize=(4*len(targets), 8))
for j,(name,f) in enumerate(targets):
    arr = np.load(f).astype(np.float32)
    cam, p = gradcam(arr)
    axes[0,j].imshow(np.where(np.isfinite(arr),arr,np.nanmean(arr)), cmap="jet")
    axes[0,j].set_title(f"{name}\nNO2 tile", fontsize=9); axes[0,j].axis("off")
    axes[1,j].imshow(np.where(np.isfinite(arr),arr,np.nanmean(arr)), cmap="gray")
    axes[1,j].imshow(cam, cmap="jet", alpha=0.5)
    axes[1,j].set_title(f"Grad-CAM  P(plant)={p:.2f}", fontsize=9); axes[1,j].axis("off")

plt.tight_layout(); plt.savefig("data/gradcam.png", dpi=120, bbox_inches="tight")
print("Saved data/gradcam.png")