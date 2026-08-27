import os

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
TRAIN_DIR       = os.path.join(DATA_DIR, "train_frames")
VAL_DIR         = os.path.join(DATA_DIR, "val_frames")
TEST_DIR        = os.path.join(DATA_DIR, "test")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")

# ── Architecture ────────────────────────────────────────────────────────────
# Increased from [16,32,64] and hidden=32.  Gives much richer feature space
# while still fitting in 4 GB VRAM (GTX 1650) at BATCH_SIZE=4.
ENCODER_CHANNELS = [32, 64, 128]
CONVLSTM_HIDDEN  = 64

# ── Input ───────────────────────────────────────────────────────────────────
# 48→64: larger spatial context reveals explosion/fire/smoke detail.
# 6→10 frames @ 5 FPS ≈ 2 s of context (vs. 1.2 s before).
IMAGE_SIZE      = 64
SEQUENCE_LENGTH = 10

# ── Training ─────────────────────────────────────────────────────────────────
# Smaller batch for GPU memory; scheduler will compensate.
BATCH_SIZE    = 4
LEARNING_RATE = 3e-4          # Will decay via CosineAnnealingLR to 1e-6
EPOCHS        = 35            # More epochs for the bigger model to converge
SAVE_EVERY    = 1

# ── Threshold calibration ────────────────────────────────────────────────────
# Lower percentiles = more sensitive to anomalies.
THRESHOLD_PERCENTILE_LOW  = 70
THRESHOLD_PERCENTILE_HIGH = 88

# Fallback hard-coded thresholds (overridden at runtime by thresholds.npy)
THRESHOLD_LOW  = 0.00015
THRESHOLD_HIGH = 0.00028

# ── DataLoader ────────────────────────────────────────────────────────────────
NUM_WORKERS = 4
PIN_MEMORY  = False           # Set True if on Linux/CUDA