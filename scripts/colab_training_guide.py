# Video Anomaly Detection — Google Colab Training Notebook
# Run this on Colab with T4 GPU (Runtime > Change runtime type > T4 GPU)
# ─────────────────────────────────────────────────────────────────────────────

# ── CELL 1: Mount Google Drive & clone / upload project ──────────────────────
"""
OPTION A — If your project is on GitHub:
    !git clone https://github.com/YOUR_USERNAME/Video-Anomaly-detection.git
    %cd Video-Anomaly-detection

OPTION B — Upload the project zip to Colab:
    from google.colab import files
    uploaded = files.upload()  # upload Video-Anomaly-detection.zip
    !unzip Video-Anomaly-detection.zip -d .
    %cd Video-Anomaly-detection

OPTION C — Mount Google Drive (recommended for large datasets):
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Video-Anomaly-detection
"""

# ── CELL 2: Install dependencies ─────────────────────────────────────────────
"""
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install opencv-python-headless tqdm matplotlib scikit-learn
"""

# ── CELL 3: Verify GPU ────────────────────────────────────────────────────────
"""
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else "N/A", "GB")
"""

# ── CELL 4: Upload your dataset ───────────────────────────────────────────────
"""
# Upload the data/ folder to Colab (or use Drive mount above).
# Your data/train/ and data/val/ must contain the .mp4 / .avi video files.
# The train_frames/ and val_frames/ will be generated in the next step.
"""

# ── CELL 5: Re-extract frames at 64×64 ───────────────────────────────────────
"""
import sys; sys.path.insert(0, '.')
from scripts.reextract_64px import clear_and_reextract
from training.config import BASE_DIR, IMAGE_SIZE
import os

DATA_DIR  = os.path.join(BASE_DIR, "data")
clear_and_reextract(os.path.join(DATA_DIR, "train"),
                    os.path.join(DATA_DIR, "train_frames"), IMAGE_SIZE, "train")
clear_and_reextract(os.path.join(DATA_DIR, "val"),
                    os.path.join(DATA_DIR, "val_frames"),   IMAGE_SIZE, "val")
"""

# ── CELL 6: Run training ──────────────────────────────────────────────────────
"""
import sys; sys.path.insert(0, '.')
from training.train import train
train()
"""

# ── CELL 7: Recompute thresholds ─────────────────────────────────────────────
"""
# After training completes, recompute thresholds on the val set
%run scripts/recompute_thresholds.py
"""

# ── CELL 8: Download checkpoints ─────────────────────────────────────────────
"""
from google.colab import files
import shutil, os

# Zip and download the checkpoints
shutil.make_archive('/content/checkpoints_new', 'zip', 'checkpoints')
files.download('/content/checkpoints_new.zip')
# Unzip on your local machine into e:/Video-Anomaly-detection/checkpoints/
"""

# ── CELL 9: Quick inference test (optional) ───────────────────────────────────
"""
# Upload a test video to Colab
from google.colab import files
uploaded = files.upload()  # upload your bombing video
video_path = list(uploaded.keys())[0]

import sys; sys.path.insert(0, '.')
from inference.detect import detect_anomalies
results, tl, th = detect_anomalies(video_path)
for r in results[:20]:
    print(f"Frame {r['frame_idx']:4d} | Score: {r['anomaly_score']:.6f} | {r['status']}")
"""

print("""
╔══════════════════════════════════════════════════════════════════╗
║           COLAB TRAINING INSTRUCTIONS                           ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Runtime > Change runtime type > T4 GPU                      ║
║  2. Upload project + dataset to Colab or mount Drive            ║
║  3. Run Cell 2  (install deps)                                  ║
║  4. Run Cell 5  (re-extract frames at 64×64)                    ║
║  5. Run Cell 6  (training — ~2-4 hours on T4)                   ║
║  6. Run Cell 7  (recompute thresholds)                          ║
║  7. Run Cell 8  (download checkpoints)                          ║
║  8. Place best_model.pt + thresholds.npy in checkpoints/        ║
╚══════════════════════════════════════════════════════════════════╝
""")
