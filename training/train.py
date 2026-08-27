import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from models.autoencoder import ConvLSTMAE
from models.dataset import VideoFrameDataset
from training.config import *


# ─────────────────────────────────────────────────────────────────────────────
# Combined reconstruction loss: MSE + L1
# MSE penalises large deviations (good for anomaly score magnitude).
# L1 preserves sharp edges / textures that MSE alone tends to blur.
# ─────────────────────────────────────────────────────────────────────────────
def combined_loss(output, target, mse_weight=0.7, l1_weight=0.3):
    mse = F.mse_loss(output, target)
    l1  = F.l1_loss(output, target)
    return mse_weight * mse + l1_weight * l1


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = VideoFrameDataset(TRAIN_DIR, SEQUENCE_LENGTH, IMAGE_SIZE,
                                      stride=3, augment=True)
    val_dataset   = VideoFrameDataset(VAL_DIR,   SEQUENCE_LENGTH, IMAGE_SIZE,
                                      stride=3, augment=False)

    print(f"Training sequences  : {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("ERROR: No training data found. Run scripts/extract_dataset_frames.py first.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = ConvLSTMAE(ENCODER_CHANNELS, CONVLSTM_HIDDEN, SEQUENCE_LENGTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=1e-5)

    # CosineAnnealingLR: smoothly decays LR from LEARNING_RATE → ~1e-6 over EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    resume_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")
    if os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device,
                                weights_only=False)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch   = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            train_losses  = checkpoint.get("train_losses", [])
            val_losses    = checkpoint.get("val_losses", [])
            print(f"Resumed at epoch {start_epoch}, "
                  f"best_val_loss so far: {best_val_loss:.6f}")
        except Exception as e:
            print(f"Could not load checkpoint weights (architecture changed?): {e}")
            print("Starting training from scratch.")
            start_epoch   = 1
            best_val_loss = float("inf")
            train_losses, val_losses = [], []

    # ── Training loop ─────────────────────────────────────────────────────────
    GRAD_ACCUM_STEPS = 2    # effective batch = BATCH_SIZE × 2 (helps GTX 1650)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch  = batch.to(device)
            output = model(batch)
            loss   = combined_loss(output, batch) / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                # Gradient clipping: prevents exploding gradients in ConvLSTM
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch  = batch.to(device)
                output = model(batch)
                loss   = combined_loss(output, batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:3d}/{EPOCHS}] | "
              f"Train: {avg_train_loss:.6f} | "
              f"Val: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.2e}")

        scheduler.step()

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        # Save latest checkpoint every epoch
        torch.save({
            'epoch':                  epoch,
            'model_state_dict':       model.state_dict(),
            'optimizer_state_dict':   optimizer.state_dict(),
            'scheduler_state_dict':   scheduler.state_dict(),
            'best_val_loss':          best_val_loss,
            'train_losses':           train_losses,
            'val_losses':             val_losses,
        }, os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt'))

        if is_best:
            torch.save({
                'epoch':              epoch,
                'model_state_dict':   model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':           best_val_loss,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
            print(f"  --> Best model saved (val_loss={best_val_loss:.6f})")

    # ── Compute anomaly thresholds from validation set ────────────────────────
    print("\nComputing anomaly thresholds from validation set...")
    model.eval()
    all_errors = []
    with torch.no_grad():
        for batch in val_loader:
            batch  = batch.to(device)
            output = model(batch)
            # Compute error ONLY on the last frame of the sequence to match detect.py
            # batch[:, -1] has shape (B, C, H, W)
            error  = torch.mean((output[:, -1] - batch[:, -1]) ** 2, dim=[1, 2, 3])
            all_errors.extend(error.cpu().numpy().flatten().tolist())

    if all_errors:
        all_errors_arr = np.array(all_errors)
        threshold_low  = np.percentile(all_errors_arr, THRESHOLD_PERCENTILE_LOW)
        threshold_high = np.percentile(all_errors_arr, THRESHOLD_PERCENTILE_HIGH)
        print(f"Error stats — "
              f"min: {all_errors_arr.min():.6f}, "
              f"mean: {all_errors_arr.mean():.6f}, "
              f"max: {all_errors_arr.max():.6f}")
        print(f"Threshold LOW  ({THRESHOLD_PERCENTILE_LOW}th pct, suspicious): "
              f"{threshold_low:.6f}")
        print(f"Threshold HIGH ({THRESHOLD_PERCENTILE_HIGH}th pct, anomaly)  : "
              f"{threshold_high:.6f}")

        np.save(os.path.join(CHECKPOINT_DIR, 'thresholds.npy'),
                np.array([threshold_low, threshold_high]))
        print("Thresholds saved to checkpoints/thresholds.npy")

    # ── Loss curve ────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Combined Loss (MSE+L1)')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'))
    print(f"Loss curve saved to outputs/loss_curve.png")
    print("\nTraining complete!")


if __name__ == "__main__":
    train()