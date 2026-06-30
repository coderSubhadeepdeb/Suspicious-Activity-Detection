import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from models.autoencoder import ConvLSTMAE
from models.dataset import VideoFrameDataset
from training.config import *


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    print(f"Using device: {device}")

    train_dataset = VideoFrameDataset(TRAIN_DIR, SEQUENCE_LENGTH, IMAGE_SIZE, stride=5)
    val_dataset   = VideoFrameDataset(VAL_DIR,   SEQUENCE_LENGTH, IMAGE_SIZE, stride=5)

    print(f"Training sequences  : {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("ERROR: No training data found. Run extract_dataset_frames.py first.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model     = ConvLSTMAE(ENCODER_CHANNELS, CONVLSTM_HIDDEN, SEQUENCE_LENGTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    start_epoch = 1
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    resume_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")
    if os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        print(f"Resumed at epoch {start_epoch}, best_val_loss so far: {best_val_loss:.6f}")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt'))

        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
            print(f"  --> Best model saved (val_loss={best_val_loss:.6f})")

    print("\nComputing anomaly thresholds from validation set...")
    model.eval()
    all_errors = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)
            error = torch.mean((output - batch) ** 2, dim=[2, 3, 4])
            all_errors.extend(error.cpu().numpy().flatten().tolist())

    if all_errors:
        all_errors_arr = np.array(all_errors)
        threshold_low  = np.percentile(all_errors_arr, 90)
        threshold_high = np.percentile(all_errors_arr, 97)
        print(f"Threshold LOW  (suspicious): {threshold_low:.6f}")
        print(f"Threshold HIGH (anomaly)   : {threshold_high:.6f}")

        np.save(os.path.join(CHECKPOINT_DIR, 'thresholds.npy'),
                np.array([threshold_low, threshold_high]))
        print("Thresholds saved to checkpoints/thresholds.npy")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'))
    print(f"Loss curve saved to outputs/loss_curve.png")
    print("\nTraining complete!")


if __name__ == "__main__":
    train()