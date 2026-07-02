import numpy as np, glob, os, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
torch.manual_seed(0); np.random.seed(0)

# ---------- load tiles ----------
def load_folder(path, label):
    X, y = [], []
    for f in sorted(glob.glob(f"{path}/*.npy")):
        arr = np.load(f).astype(np.float32)
        # fill cloud-gap NaNs with the tile's own mean (simple, safe)
        m = np.nanmean(arr)
        arr = np.where(np.isfinite(arr), arr, m)
        X.append(arr); y.append(label)
    return X, y

Xp, yp = load_folder("data/monthly/positive", 1)   # plume
Xn, yn = load_folder("data/monthly/negative", 0)   # no-plume
X = np.stack(Xp + Xn)[:, None, :, :]               # shape (N,1,64,64)
y = np.array(yp + yn, dtype=np.int64)
print(f"loaded {len(y)} tiles  (plume={sum(y==1)}, no-plume={sum(y==0)})")

# ---------- normalize (per-dataset) ----------
mean, std = X.mean(), X.std() + 1e-12
X = (X - mean) / std

X = torch.tensor(X); y = torch.tensor(y)
ds = TensorDataset(X, y)
n_test = max(1, int(0.2 * len(ds)))
n_train = len(ds) - n_test
train_ds, test_ds = random_split(ds, [n_train, n_test],
                                 generator=torch.Generator().manual_seed(0))
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=16)
print(f"train={n_train}  test={n_test}")

# ---------- tiny CNN classifier ----------
class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64, 2))
    def forward(self, x): return self.net(x)

model = Detector().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
lossf = nn.CrossEntropyLoss()

# ---------- train ----------
EPOCHS = 30
for ep in range(EPOCHS):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward(); opt.step()

    # eval each epoch
    model.eval(); correct = total = 0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item(); total += yb.size(0)
    if ep % 5 == 0 or ep == EPOCHS-1:
        print(f"epoch {ep:2d}  loss {loss.item():.3f}  test-acc {100*correct/total:.1f}%")

print(f"\nFinal test accuracy: {100*correct/total:.1f}%  (chance = 50%)")
torch.save(model.state_dict(), "detector.pt")
print("Saved detector.pt")