"""
Re-extract training & validation frames at the new IMAGE_SIZE (64×64).

The existing train_frames/ and val_frames/ were created at 48×48.
The new model requires 64×64 — run this script ONCE before training.

Usage:
    python scripts/reextract_64px.py
"""
import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.extract_dataset_frames import extract_frames_from_videos
from training.config import BASE_DIR, IMAGE_SIZE

DATA_DIR   = os.path.join(BASE_DIR, "data")
TRAIN_SRC  = os.path.join(DATA_DIR, "train")
VAL_SRC    = os.path.join(DATA_DIR, "val")
TRAIN_OUT  = os.path.join(DATA_DIR, "train_frames")
VAL_OUT    = os.path.join(DATA_DIR, "val_frames")


def clear_and_reextract(src_dir, out_dir, image_size, label):
    if not os.path.isdir(src_dir):
        print(f"[SKIP] {label} source dir not found: {src_dir}")
        return

    # Wipe old frames
    if os.path.isdir(out_dir):
        print(f"Clearing old {label} frames from {out_dir} ...")
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Extracting {label} at {image_size}×{image_size} ...")
    extract_frames_from_videos(src_dir, out_dir,
                               image_size=image_size, target_fps=5)
    print(f"Done — {label} frames saved to {out_dir}")


if __name__ == "__main__":
    print(f"Re-extracting at IMAGE_SIZE = {IMAGE_SIZE}px\n")
    clear_and_reextract(TRAIN_SRC, TRAIN_OUT, IMAGE_SIZE, "train")
    clear_and_reextract(VAL_SRC,   VAL_OUT,   IMAGE_SIZE, "val")
    print("\nAll frames re-extracted. You can now run: python training/train.py")
