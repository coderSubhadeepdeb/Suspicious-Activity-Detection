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

model = ConvLSTMAE(ENCODER_CHANNELS, CONVLSTM_HIDDEN, SEQUENCE_LENGTH).to(device)
checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

val_dataset = VideoFrameDataset(VAL_DIR, SEQUENCE_LENGTH, IMAGE_SIZE, stride=5)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

all_errors = []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        output = model(batch)
        error = torch.mean((output - batch) ** 2, dim=[2, 3, 4])
        all_errors.extend(error.cpu().numpy().flatten().tolist())

all_errors_arr = np.array(all_errors)
threshold_low  = np.percentile(all_errors_arr, 90)
threshold_high = np.percentile(all_errors_arr, 97)

print(f"Old thresholds — LOW: 0.016775, HIGH: 0.021008")
print(f"New thresholds — LOW: {threshold_low:.6f}, HIGH: {threshold_high:.6f}")
print(f"Error distribution — min: {all_errors_arr.min():.6f}, max: {all_errors_arr.max():.6f}, mean: {all_errors_arr.mean():.6f}")

np.save(os.path.join(CHECKPOINT_DIR, "thresholds.npy"), np.array([threshold_low, threshold_high]))
print("Thresholds updated and saved.")