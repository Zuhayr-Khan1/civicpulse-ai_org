# CivicPulse India — Notebooks 04-07 (Dual-Mode Edition)
*Updated: 2026-02-21 | Supports `LOAD_MODE = "hdf5"` (low-RAM) or `LOAD_MODE = "normal"` (high-end PC)*

---

## Verification: Notebooks 00-03 Status

| # | Notebook | Key Changes Confirmed |
|---|----------|-----------------------|
| 00 | Setup_India_Boundaries | ✅ Exports `data/raw/india_regions.geojson` + `.shp`; map saved to `data/viz/` |
| 01 | Preprocess_sample_States | ✅ Loops `worldpop_files` directly (no `available_years`); nodata=-99999 masked to 0; keys are full stems e.g. `ind_ppp_2000_1km_Aggregated` |
| 02 | Create_HDF5_Dataset | ✅ Pads Telangana/Maharashtra to same spatial dims; writes `data/processed/india_sample.h5` |
| 03 | Clip_Full_India | ✅ Uses `re.search` year extraction from filenames; tqdm progress; logging to `logs/` |

---

## 🔧 DUAL-MODE SWITCH — set this in Cell 1 of EACH notebook below

```python
# ============================================================
#  SET THIS VARIABLE TO SWITCH MODES
#  "hdf5"   → lazy HDF5 loading, low RAM usage  (default)
#  "normal" → full numpy loading, high-end PC
# ============================================================
LOAD_MODE = "hdf5"   # <-- change to "normal" if you have 32GB+ RAM
```

---

---

# NOTEBOOK 04 — Model Architecture

```python
# Cell 1 — Imports + Mode Switch
import sys, os, torch, torch.nn as nn, torch.optim as optim
import numpy as np, h5py, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import warnings; warnings.filterwarnings("ignore")
from src.config import TrainingConfig

# ============================================================
LOAD_MODE = "hdf5"   # "hdf5" or "normal"
# ============================================================

config = TrainingConfig()
device = torch.device(config.DEVICE)

print("=" * 70)
print("NOTEBOOK 04 — MODEL ARCHITECTURE")
print("=" * 70)
print(f"  Mode     : {LOAD_MODE}")
print(f"  Device   : {device}")
print(f"  Batch    : {config.BATCH_SIZE}")
print(f"  LR       : {config.LEARNING_RATE}")
```

```python
# Cell 2 — ConvLSTM Cell
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv_gates = nn.Conv2d(
            in_channels + hidden_channels, 2 * hidden_channels, kernel_size, padding=padding)
        self.conv_candidate = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, inputs, hidden_state):
        h, c = hidden_state
        combined = torch.cat([inputs, h], dim=1)
        gates = self.conv_gates(combined)
        reset_gate, update_gate = torch.split(gates, self.hidden_channels, dim=1)
        reset_gate  = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        combined_candidate = torch.cat([inputs, reset_gate * h], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_candidate))
        new_c = (1 - update_gate) * c + update_gate * candidate
        new_h = torch.tanh(new_c) * update_gate + (1 - update_gate) * h
        return new_h, new_c
```

```python
# Cell 3 — ConvLSTM Encoder-Decoder
class ConvLSTMEncoderDecoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, num_layers=2, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.encoder_cells = nn.ModuleList([
            ConvLSTMCell(in_channels if i == 0 else hidden_channels, hidden_channels, kernel_size)
            for i in range(num_layers)])
        self.decoder_cells = nn.ModuleList([
            ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
            for _ in range(num_layers)])
        self.output_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (batch, time, channels, H, W)
        B, T, C, H, W = x.shape
        h = [torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
             for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
             for _ in range(self.num_layers)]
        for t in range(T):
            xt = x[:, t]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.encoder_cells[layer](xt if layer == 0 else h[layer-1], (h[layer], c[layer]))
        for layer in range(self.num_layers):
            inp = h[layer-1] if layer > 0 else h[0]
            h[layer], c[layer] = self.decoder_cells[layer](inp, (h[layer], c[layer]))
        return self.output_conv(h[-1])   # (B, 1, H, W)
```

```python
# Cell 4 — Load Test Data (mode-aware)
h5_path = "data/processed/india_sample.h5"

if LOAD_MODE == "hdf5":
    print("📂 HDF5 mode: lazy-loading patch for test...")
    with h5py.File(h5_path, "r") as h5:
        data = h5["population_data"][:, :256, :256]   # (5, 256, 256)
    print(f"  Loaded patch: {data.shape}")
else:
    print("📂 Normal mode: full numpy load...")
    tel_seq  = np.load("data/processed/telangana_population_sequence.npy")
    maha_seq = np.load("data/processed/maharashtra_population_sequence.npy")
    # Pad to same dims
    T, H1, W1 = tel_seq.shape
    _,  H2, W2 = maha_seq.shape
    maxH, maxW = max(H1, H2), max(W1, W2)
    tel_seq  = np.pad(tel_seq,  ((0,0),(0,maxH-H1),(0,maxW-W1)))
    maha_seq = np.pad(maha_seq, ((0,0),(0,maxH-H2),(0,maxW-W2)))
    data = np.concatenate([tel_seq, maha_seq], axis=1)[:, :256, :256]
    print(f"  Loaded shape: {data.shape}")

X_test = torch.from_numpy(data[:4]).float().unsqueeze(0).unsqueeze(2)  # (1,4,1,256,256)
y_test = torch.from_numpy(data[4]).float().unsqueeze(0).unsqueeze(0)   # (1,1,256,256)

model = ConvLSTMEncoderDecoder().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

with torch.no_grad():
    output = model(X_test.to(device))
print(f"Output shape : {output.shape}")
print(f"Output range : {output.min().item():.1f} – {output.max().item():.1f}")
```

```python
# Cell 5 — Loss + Metrics
class PopulationLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        mae = self.mae(pred, target)
        pc  = torch.clamp(pred,   min=0)
        tc  = torch.clamp(target, min=0)
        mask = tc > 1.0
        rel  = (torch.abs(pc[mask] - tc[mask]) / (tc[mask] + 1e-8)).mean() if mask.sum() > 0                else torch.tensor(0.0, device=pred.device)
        return self.alpha * mse + self.beta * mae + 0.1 * rel

def calculate_r2(pred, target):
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()

criterion = PopulationLoss()
y_dev = y_test.to(device)
loss  = criterion(output, y_dev)
r2    = calculate_r2(output, y_dev)
print(f"Loss: {loss.item():.4f}  |  R²: {r2:.3f}")
```

```python
# Cell 6 — Save architecture
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/model_architecture.pt")
print("✅ Saved: models/model_architecture.pt")
print("=" * 70)
print("NOTEBOOK 04 COMPLETE — Next: Notebook 05 Training")
print("=" * 70)
```

---

---

# NOTEBOOK 05 — Progressive Training

```python
# Cell 1 — Imports + Mode Switch
import sys, os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, h5py, json, time, logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import warnings; warnings.filterwarnings("ignore")
from src.config import TrainingConfig

# ============================================================
LOAD_MODE = "hdf5"   # "hdf5" or "normal"
# ============================================================

config = TrainingConfig()
device = torch.device(config.DEVICE)
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(filename="logs/training.log", level=logging.INFO,
                    format="%(asctime)s - %(message)s")

print("=" * 70)
print("NOTEBOOK 05 — PROGRESSIVE TRAINING")
print("=" * 70)
print(f"  Mode: {LOAD_MODE}  |  Device: {device}")
```

```python
# Cell 2 — Dataset Classes (HDF5 and Normal)

class PopulationDatasetHDF5(Dataset):
    """Lazy HDF5 dataset — low RAM."""
    def __init__(self, h5_path, patch_size=64, stride=32, downsample=1):
        self.h5_path    = h5_path
        self.patch_size = patch_size
        self.stride     = stride
        self.downsample = downsample
        with h5py.File(h5_path, "r") as h5:
            shape = h5["population_data"].shape
        self.height = shape[1] // downsample
        self.width  = shape[2] // downsample
        self.patches = [(y, x)
                        for y in range(0, self.height - patch_size, stride)
                        for x in range(0, self.width  - patch_size, stride)]

    def __len__(self): return len(self.patches)

    def __getitem__(self, idx):
        y, x = self.patches[idx]
        ds   = self.downsample
        ps   = self.patch_size
        with h5py.File(self.h5_path, "r") as h5:
            data = h5["population_data"][:,
                       y*ds : (y+ps)*ds : ds,
                       x*ds : (x+ps)*ds : ds]   # (T, ps, ps)
        X = torch.from_numpy(data[:4].copy()).float().unsqueeze(1)   # (4,1,H,W)
        y_ = torch.from_numpy(data[4].copy()).float().unsqueeze(0)   # (1,H,W)
        return X, y_


class PopulationDatasetNormal(Dataset):
    """Full in-memory dataset — high-end PC."""
    def __init__(self, data_array, patch_size=64, stride=32):
        # data_array: (T, H, W) numpy
        self.data       = data_array
        self.patch_size = patch_size
        T, H, W = data_array.shape
        self.patches = [(y, x)
                        for y in range(0, H - patch_size, stride)
                        for x in range(0, W - patch_size, stride)]

    def __len__(self): return len(self.patches)

    def __getitem__(self, idx):
        y, x = self.patches[idx]
        ps   = self.patch_size
        data = self.data[:, y:y+ps, x:x+ps]
        X  = torch.from_numpy(data[:4].copy()).float().unsqueeze(1)
        y_ = torch.from_numpy(data[4].copy()).float().unsqueeze(0)
        return X, y_


def make_dataset(h5_path, normal_data, patch_size, stride, downsample):
    """Factory: returns correct dataset based on LOAD_MODE."""
    if LOAD_MODE == "hdf5":
        return PopulationDatasetHDF5(h5_path, patch_size, stride, downsample)
    else:
        # normal_data already full array (T,H,W)
        return PopulationDatasetNormal(normal_data, patch_size, stride)
```

```python
# Cell 3 — Load Data (mode-aware)
h5_path = "data/processed/india_sample.h5"
normal_data = None

if LOAD_MODE == "hdf5":
    print("📂 HDF5 mode — data will be loaded lazily per patch.")
    with h5py.File(h5_path, "r") as h5:
        print(f"  Dataset shape: {h5['population_data'].shape}")
else:
    print("📂 Normal mode — loading full arrays into RAM...")
    tel  = np.load("data/processed/telangana_population_sequence.npy")
    maha = np.load("data/processed/maharashtra_population_sequence.npy")
    T, H1, W1 = tel.shape
    _,  H2, W2 = maha.shape
    maxH, maxW = max(H1,H2), max(W1,W2)
    tel  = np.pad(tel,  ((0,0),(0,maxH-H1),(0,maxW-W1)))
    maha = np.pad(maha, ((0,0),(0,maxH-H2),(0,maxW-W2)))
    normal_data = np.concatenate([tel, maha], axis=1).astype(np.float32)
    print(f"  Full array loaded: {normal_data.shape}")
```

```python
# Cell 4 — Model classes (copy from NB04)
# [Paste ConvLSTMCell, ConvLSTMEncoderDecoder, PopulationLoss, calculate_r2 here]
# (same code as Notebook 04 Cells 2, 3, 5)
```

```python
# Cell 5 — Training Stages Config
training_stages = {
    "stage1_coarse": dict(
        downsample=4, patch_size=32,
        batch_size=16 if "cuda" in str(device) else 4,
        epochs=20, lr=1e-3, name="Stage 1 — Coarse"),
    "stage2_medium": dict(
        downsample=2, patch_size=64,
        batch_size=8  if "cuda" in str(device) else 2,
        epochs=50, lr=5e-4, name="Stage 2 — Medium"),
    "stage3_fine":   dict(
        downsample=1, patch_size=128,
        batch_size=4  if "cuda" in str(device) else 1,
        epochs=100, lr=1e-4, name="Stage 3 — Fine"),
}
for s, cfg in training_stages.items():
    print(f"{cfg['name']}  |  ds={cfg['downsample']}  "
          f"patch={cfg['patch_size']}  bs={cfg['batch_size']}  ep={cfg['epochs']}")
```

```python
# Cell 6 — Trainer Class
class Trainer:
    def __init__(self, model, device, checkpoint_dir="models/checkpoints"):
        self.model = model.to(device)
        self.device = device
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_r2 = -np.inf

    def train_epoch(self, loader, criterion, optimizer):
        self.model.train()
        total_loss = total_r2 = 0
        for X, y in tqdm(loader, desc="Train", leave=False):
            X, y = X.to(self.device), y.to(self.device)
            out  = self.model(X)
            loss = criterion(out, y)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_r2   += calculate_r2(out.detach(), y.detach())
        n = len(loader)
        return total_loss / n, total_r2 / n

    def validate(self, loader, criterion):
        self.model.eval()
        total_loss = total_r2 = 0
        with torch.no_grad():
            for X, y in tqdm(loader, desc="Val ", leave=False):
                X, y = X.to(self.device), y.to(self.device)
                out  = self.model(X)
                total_loss += criterion(out, y).item()
                total_r2   += calculate_r2(out, y)
        n = len(loader)
        return total_loss / n, total_r2 / n

    def save_checkpoint(self, epoch, stage, loss, r2, is_best=False):
        ckpt = dict(epoch=epoch, stage=stage,
                    model_state=self.model.state_dict(), loss=loss, r2=r2)
        path = self.ckpt_dir / f"ckpt_{stage}_ep{epoch}.pt"
        torch.save(ckpt, path)
        if is_best:
            torch.save(ckpt, self.ckpt_dir / "best_model.pt")
            print(f"  ⭐ New best R²={r2:.3f} — saved best_model.pt")
        return path
```

```python
# Cell 7 — Progressive Training Loop
model   = ConvLSTMEncoderDecoder().to(device)
trainer = Trainer(model, device)
criterion = PopulationLoss()
all_history = {}

for stage_name, cfg in training_stages.items():
    print("=" * 70)
    print(cfg["name"])
    print("=" * 70)

    dataset = make_dataset(
        h5_path, normal_data,
        patch_size=cfg["patch_size"],
        stride=cfg["patch_size"] // 2,
        downsample=cfg["downsample"])

    train_sz = int(0.8 * len(dataset))
    val_sz   = len(dataset) - train_sz
    train_set, val_set = torch.utils.data.random_split(dataset, [train_sz, val_sz])

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True)

    history  = dict(train_loss=[], train_r2=[], val_loss=[], val_r2=[])
    best_r2  = -np.inf

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_r2 = trainer.train_epoch(train_loader, criterion, optimizer)
        vl_loss, vl_r2 = trainer.validate(val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["train_r2"].append(tr_r2)
        history["val_loss"].append(vl_loss)
        history["val_r2"].append(vl_r2)

        is_best = vl_r2 > best_r2
        if is_best: best_r2 = vl_r2
        trainer.save_checkpoint(epoch, stage_name, vl_loss, vl_r2, is_best=is_best)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{cfg['epochs']} "
                  f"| Train L={tr_loss:.4f} R²={tr_r2:.3f} "
                  f"| Val L={vl_loss:.4f} R²={vl_r2:.3f}")

        scheduler.step(vl_r2)
        logging.info(f"{stage_name} ep{epoch} tr={tr_loss:.4f}/{tr_r2:.3f} "
                     f"val={vl_loss:.4f}/{vl_r2:.3f}")

    all_history[stage_name] = history
    print(f"✅ {cfg['name']} done | Best Val R²={best_r2:.3f}")

with open("logs/training_summary.json", "w") as f:
    json.dump({k: {kk: vv[-1] for kk, vv in v.items()} for k, v in all_history.items()}, f, indent=2)
print("\n✅ Training summary → logs/training_summary.json")
print("✅ Best model      → models/checkpoints/best_model.pt")
```

---

---

# NOTEBOOK 06 — Inference & Predictions

```python
# Cell 1 — Imports + Mode Switch
import sys, os, torch, numpy as np, rasterio, rasterio.transform, h5py
from pathlib import Path
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
from src.config import TrainingConfig

# ============================================================
LOAD_MODE = "hdf5"   # "hdf5" or "normal"
# ============================================================

config = TrainingConfig()
device = torch.device(config.DEVICE)

print("=" * 70)
print("NOTEBOOK 06 — INFERENCE & PREDICTIONS")
print("=" * 70)
print(f"  Mode: {LOAD_MODE}  |  Device: {device}")
```

```python
# Cell 2 — Load Model
# [Paste ConvLSTMCell + ConvLSTMEncoderDecoder here — same as NB04]

model_path = "models/checkpoints/best_model.pt"
model = ConvLSTMEncoderDecoder().to(device)
if Path(model_path).exists():
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"✅ Loaded checkpoint: {model_path}  (R²={ckpt.get('r2', 'N/A')})")
else:
    print("⚠️  No checkpoint found — using untrained weights (demo only)")
model.eval()
```

```python
# Cell 3 — Load Data (mode-aware)
h5_path = "data/processed/india_sample.h5"

if LOAD_MODE == "hdf5":
    print("📂 HDF5 mode — loading data lazily per prediction call...")
    with h5py.File(h5_path, "r") as h5:
        data = h5["population_data"][:]          # load all 5 timesteps fully for inference
        metadata = {k: h5.attrs[k] for k in h5.attrs}
else:
    print("📂 Normal mode — loading full numpy arrays...")
    tel  = np.load("data/processed/telangana_population_sequence.npy")
    maha = np.load("data/processed/maharashtra_population_sequence.npy")
    T, H1, W1 = tel.shape
    _,  H2, W2 = maha.shape
    maxH, maxW = max(H1,H2), max(W1,W2)
    tel  = np.pad(tel,  ((0,0),(0,maxH-H1),(0,maxW-W1)))
    maha = np.pad(maha, ((0,0),(0,maxH-H2),(0,maxW-W2)))
    data = np.concatenate([tel, maha], axis=1).astype(np.float32)
    metadata = {"years": "2000,2005,2010,2015,2020"}

print(f"  Data shape: {data.shape}  |  {metadata}")
```

```python
# Cell 4 — Patch-based Prediction Function
def predict_full(model, data_sequence, patch_size=512, overlap=64, device="cpu"):
    T, H, W = data_sequence.shape
    output   = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    stride = patch_size - overlap

    patches = [(y, x)
               for y in range(0, H - patch_size + 1, stride)
               for x in range(0, W - patch_size + 1, stride)]

    with torch.no_grad():
        for y, x in tqdm(patches, desc="Predicting patches"):
            patch = data_sequence[:, y:y+patch_size, x:x+patch_size]
            X = torch.from_numpy(patch[:4].copy()).float().unsqueeze(0).unsqueeze(2).to(device)
            pred = model(X).squeeze().cpu().numpy()
            output[y:y+patch_size, x:x+patch_size]    += pred
            count_map[y:y+patch_size, x:x+patch_size] += 1

    return output / (count_map + 1e-8)
```

```python
# Cell 5 — Generate 2025 & 2030 Predictions
predictions = {}

print("🔮 Predicting 2025 (input: 2005–2020)...")
data_2005_2020 = data[1:]                      # years 2005,2010,2015,2020
pred_2025 = predict_full(model, data_2005_2020, device=device)
predictions["2025"] = pred_2025
print(f"  Shape: {pred_2025.shape}  Range: {pred_2025.min():.0f}–{pred_2025.max():.0f}")

print("\n🔮 Predicting 2030 (input: 2010–2025)...")
data_2010_2025 = np.concatenate([data[2:], pred_2025[np.newaxis]], axis=0)  # 2010,2015,2020,2025
pred_2030 = predict_full(model, data_2010_2025, device=device)
predictions["2030"] = pred_2030
print(f"  Shape: {pred_2030.shape}  Range: {pred_2030.min():.0f}–{pred_2030.max():.0f}")
```

```python
# Cell 6 — Save as GeoTIFF
proj_dir = Path("data/projections")
proj_dir.mkdir(parents=True, exist_ok=True)

ref_tifs = list(Path("data/processed").glob("india_pop_clipped_*.tif"))
base_profile = None
if ref_tifs:
    with rasterio.open(ref_tifs[0]) as src:
        base_profile = src.profile
else:
    base_profile = dict(
        driver="GTiff", dtype="float32", nodata=None,
        width=pred_2025.shape[1], height=pred_2025.shape[0],
        count=1, crs="EPSG:4326",
        transform=rasterio.transform.Affine(1, 0, 72, 0, -1, 35))

for year, pred in predictions.items():
    out_path = proj_dir / f"population_prediction_{year}.tif"
    base_profile.update(dtype="float32", width=pred.shape[1], height=pred.shape[0], count=1)
    with rasterio.open(out_path, "w", **base_profile) as dst:
        dst.write(pred, 1)
    print(f"  ✅ Saved {out_path}")
```

```python
# Cell 7 — Validation vs Actual 2020
actual_2020 = data[4]
pred_2020   = predict_full(model, data[:4], device=device)

mae  = np.abs(pred_2020 - actual_2020).mean()
rmse = np.sqrt(((pred_2020 - actual_2020) ** 2).mean())
ss_res = ((pred_2020 - actual_2020) ** 2).sum()
ss_tot = ((actual_2020 - actual_2020.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot

print("=" * 70)
print("PREDICTION VALIDATION (2020 actual vs predicted)")
print("=" * 70)
print(f"  R²  : {r2:.3f}")
print(f"  MAE : {mae:.1f} people/km²")
print(f"  RMSE: {rmse:.1f} people/km²")
print("\n✅ Next: Notebook 07 — Gap Analysis")
```

---

---

# NOTEBOOK 07 — Infrastructure Gap Analysis

```python
# Cell 1 — Imports + Mode Switch
import sys, os, numpy as np, rasterio, geopandas as gpd, pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
from src.region_manager import ConfigurableBoundaryManager

# ============================================================
LOAD_MODE = "hdf5"   # "hdf5" or "normal"
#   NOTE: Gap analysis doesn't use HDF5 directly — predictions
#   are loaded from GeoTIFF. This switch kept for consistency.
# ============================================================

print("=" * 70)
print("NOTEBOOK 07 — INFRASTRUCTURE GAP ANALYSIS")
print("=" * 70)
```

```python
# Cell 2 — Load Predictions
pred_2025 = rasterio.open("data/projections/population_prediction_2025.tif").read(1)
pred_2030 = rasterio.open("data/projections/population_prediction_2030.tif").read(1)
print(f"2025: {pred_2025.shape}  range {pred_2025.min():.0f}–{pred_2025.max():.0f}")
print(f"2030: {pred_2030.shape}  range {pred_2030.min():.0f}–{pred_2030.max():.0f}")
```

```python
# Cell 3 — Infrastructure Standards
standards = {
    "hospitals_primary":   dict(people_per_facility=50_000,  name="Primary Health Centres"),
    "hospitals_secondary": dict(people_per_facility=500_000, name="Secondary Hospitals"),
    "schools":             dict(people_per_facility=3_000,   name="Schools"),
    "water_stations":      dict(people_per_facility=10_000,  name="Water Supply Stations"),
    "police_stations":     dict(people_per_facility=100_000, name="Police Stations"),
}
print("=" * 70)
print("INFRASTRUCTURE DENSITY STANDARDS (WHO / World Bank)")
print("=" * 70)
for k, v in standards.items():
    print(f"  {v['name']:30s}  1 per {v['people_per_facility']:>10,} people")
```

```python
# Cell 4 — Per-region Requirements
mgr     = ConfigurableBoundaryManager()
regions = ["Telangana", "Maharashtra"]
rows    = []

for rname in regions:
    region = mgr.get_region(rname)
    for year, pred in [("2025", pred_2025), ("2030", pred_2030)]:
        pop = float(pred.sum())
        row = dict(Region=rname, Year=year, Population=pop)
        for k, v in standards.items():
            row[v["name"]] = int(pop / v["people_per_facility"])
        rows.append(row)
        print(f"  {rname} {year}: pop={pop:,.0f}")

df = pd.DataFrame(rows)
print("\n", df.to_string(index=False))
```

```python
# Cell 5 — National-level + Growth Analysis
total_2025 = float(pred_2025.sum())
total_2030 = float(pred_2030.sum())
growth_pct = (total_2030 - total_2025) / total_2025 * 100

print(f"Total 2025: {total_2025:>15,.0f}")
print(f"Total 2030: {total_2030:>15,.0f}")
print(f"Growth    : {growth_pct:.2f}%")

growth_rate   = (pred_2030 - pred_2025) / (pred_2025 + 1e-8) * 100
high_growth   = growth_rate > 5.0
print(f"High-growth cells (>5% ann.): {high_growth.sum():,} ({high_growth.mean()*100:.1f}%)")
```

```python
# Cell 6 — Gap Analysis Report + Save
report_rows = []
for year, pop in [("2025", total_2025), ("2030", total_2030)]:
    row = dict(Region="All India", Year=year, Population=pop)
    for k, v in standards.items():
        row[v["name"]] = int(pop / v["people_per_facility"])
    report_rows.append(row)

report_df = pd.DataFrame(report_rows)
report_path = Path("data/projections/gap_analysis_report.csv")
report_path.parent.mkdir(parents=True, exist_ok=True)
report_df.to_csv(report_path, index=False)
print(f"✅ Report saved: {report_path}")
print(report_df.to_string(index=False))
```

```python
# Cell 7 — Recommendations
recs = []
for k, v in standards.items():
    add = int(total_2030/v["people_per_facility"]) - int(total_2025/v["people_per_facility"])
    if add > 0:
        recs.append(dict(
            Facility=v["name"],
            Additional=add,
            Priority="HIGH" if add > 100 else "MEDIUM"))

rec_df = pd.DataFrame(recs).sort_values("Additional", ascending=False)

rec_path = Path("data/projections/recommendations.csv")
rec_df.to_csv(rec_path, index=False)
print("=" * 70)
print("INFRASTRUCTURE INVESTMENT PRIORITIES 2025–2030")
print("=" * 70)
print(rec_df.to_string(index=False))
print(f"\n✅ Recommendations saved: {rec_path}")
```

```python
# Cell 8 — Complete
print("=" * 70)
print("GAP ANALYSIS COMPLETE")
print("=" * 70)
print(f"  Population growth 2025→2030: {total_2030 - total_2025:,.0f} people")
print("  Outputs:")
print("    data/projections/gap_analysis_report.csv")
print("    data/projections/recommendations.csv")
print("    data/projections/population_prediction_2025.tif")
print("    data/projections/population_prediction_2030.tif")
print("\nNext: Notebook 08 — Dashboard & Deployment")
```

---

*End of Notebooks_04-07_DualMode.md*
