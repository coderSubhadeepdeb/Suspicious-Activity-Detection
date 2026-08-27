"""
Recompute anomaly thresholds from the validation set using the current
trained model and save them to checkpoints/thresholds.npy.

Run this after training (or whenever you want to recalibrate sensitivity):
    python scripts/recompute_thresholds.py

Percentiles (from config):
    LOW  = 80th pct  → suspicious  (was 90th — more sensitive)
    HIGH = 93rd pct  → anomaly     (was 97th — more sensitive)
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader

from models.autoencoder import ConvLSTMAE
from models.dataset import VideoFrameDataset
from training.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = ConvLSTMAE(ENCODER_CHANNELS, CONVLSTM_HIDDEN, SEQUENCE_LENGTH).to(device)
ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model loaded from {ckpt_path}")

# Load validation set (no augmentation)
val_dataset = VideoFrameDataset(VAL_DIR, SEQUENCE_LENGTH, IMAGE_SIZE,
                                stride=3, augment=False)
val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)

print(f"Running inference on {len(val_dataset)} val sequences...")
all_errors = []
with torch.no_grad():
    for batch in val_loader:
        batch  = batch.to(device)
        output = model(batch)
        # Compute error ONLY on the last frame of the sequence to match detect.py
        # batch[:, -1] has shape (B, C, H, W)
        error  = torch.mean((output[:, -1] - batch[:, -1]) ** 2, dim=[1, 2, 3])
        all_errors.extend(error.cpu().numpy().flatten().tolist())

all_errors_arr = np.array(all_errors)
threshold_low  = np.percentile(all_errors_arr, THRESHOLD_PERCENTILE_LOW)
threshold_high = np.percentile(all_errors_arr, THRESHOLD_PERCENTILE_HIGH)

print("\n── Error distribution ──────────────────────────────────────────")
print(f"  min   : {all_errors_arr.min():.6f}")
print(f"  5th   : {np.percentile(all_errors_arr, 5):.6f}")
print(f"  25th  : {np.percentile(all_errors_arr, 25):.6f}")
print(f"  50th  : {np.percentile(all_errors_arr, 50):.6f}")
print(f"  75th  : {np.percentile(all_errors_arr, 75):.6f}")
print(f"  80th  : {np.percentile(all_errors_arr, 80):.6f}  ← suspicious threshold")
print(f"  90th  : {np.percentile(all_errors_arr, 90):.6f}")
print(f"  93rd  : {np.percentile(all_errors_arr, 93):.6f}  ← anomaly threshold")
print(f"  99th  : {np.percentile(all_errors_arr, 99):.6f}")
print(f"  max   : {all_errors_arr.max():.6f}")
print(f"  mean  : {all_errors_arr.mean():.6f}")
print(f"  std   : {all_errors_arr.std():.6f}")
print("────────────────────────────────────────────────────────────────")
print(f"\nNew threshold LOW  ({THRESHOLD_PERCENTILE_LOW}th pct): {threshold_low:.6f}")
print(f"New threshold HIGH ({THRESHOLD_PERCENTILE_HIGH}th pct): {threshold_high:.6f}")

np.save(os.path.join(CHECKPOINT_DIR, "thresholds.npy"),
        np.array([threshold_low, threshold_high]))
print("\nThresholds saved to checkpoints/thresholds.npy  ✓")