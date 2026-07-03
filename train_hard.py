import numpy as np, glob, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
torch.manual_seed(0); np.random.seed(0)

def load_folder(path, label):
    X, y = [], []
    for f in sorted(glob.glob(f"{path}/*.npy")):
        arr = np.load(f).astype(np.float32)
        arr = np.where(np.isfinite(arr), arr, np.nanmean(arr))  # fill gaps
        X.append(arr); y.append(label)
    return X, y

# load all three pools
Xp, yp = load_folder("data/monthly/positive",      1)   # plants
Xh, yh = load_folder("data/monthly/hard_negative",  0)  # cities/industry/hwy
Xr, yr = load_folder("data/monthly/negative",       0)  # rural

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

def run(Xlist, ylist, tag, epochs=30):
    X = np.stack(Xlist)[:, None, :, :]
    y = np.array(ylist, dtype=np.int64)
    mean, std = X.mean(), X.std() + 1e-12
    X = (X - mean) / std
    X = torch.tensor(X); y = torch.tensor(y)
    ds = TensorDataset(X, y)
    n_test = max(1, int(0.2*len(ds))); n_train = len(ds)-n_test
    tr, te = random_split(ds, [n_train, n_test],
                          generator=torch.Generator().manual_seed(0))
    tr_dl = DataLoader(tr, batch_size=16, shuffle=True)
    te_dl = DataLoader(te, batch_size=16)

    model = Detector().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()

    model.eval(); correct = total = 0
    # also track per-class so we see WHICH side it fails on
    tp=tn=fp=fn=0
    with torch.no_grad():
        for xb, yb in te_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(1)
            correct += (pred==yb).sum().item(); total += yb.size(0)
            tp += ((pred==1)&(yb==1)).sum().item()
            tn += ((pred==0)&(yb==0)).sum().item()
            fp += ((pred==1)&(yb==0)).sum().item()
            fn += ((pred==0)&(yb==1)).sum().item()
    acc = 100*correct/total
    print(f"\n=== {tag} ===")
    print(f"  train={n_train} test={n_test}")
    print(f"  test accuracy: {acc:.1f}%   (chance = 50%)")
    print(f"  plants correct (recall): {100*tp/max(tp+fn,1):.0f}%   "
          f"negatives correct: {100*tn/max(tn+fp,1):.0f}%")
    print(f"  false alarms (neg->plant): {fp}   missed plants: {fn}")
    torch.save(model.state_dict(), f"detector_{tag}.pt")
    return acc

# --- Version 1: plants vs HARD negatives only ---
# balance it: 120 plants vs 120 hard-neg (sample) for a fair comparison
import random; random.seed(0)
idx = list(range(len(Xh))); random.shuffle(idx); idx = idx[:len(Xp)]
Xh_bal = [Xh[i] for i in idx]; yh_bal = [0]*len(Xh_bal)
run(Xp + Xh_bal, yp + yh_bal, "hard_only")

# --- Version 2: plants vs ALL negatives (rural + hard) ---
run(Xp + Xh + Xr, yp + yh + yr, "mixed")

print("\nSaved detector_hard_only.pt and detector_mixed.pt")