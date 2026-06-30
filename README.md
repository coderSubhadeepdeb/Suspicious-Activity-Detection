# SentinelAI — Video Anomaly Detection System

An AI-powered suspicious activity detection system that trains on normal surveillance footage and automatically flags unusual activities (fights, explosions, robberies, assaults, arson) with a **red/yellow/green heatmap overlay**.

![SentinelAI](https://img.shields.io/badge/AI-Anomaly%20Detection-red) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)

---

## What It Does

- Trains **only on normal surveillance footage** — no labelled anomalies needed
- At inference, anything that deviates from "normal" gets flagged automatically
- Outputs a **red/yellow/green heatmap overlay** on every frame:
  - 🟢 **Green** — Normal activity
  - 🟡 **Yellow** — Suspicious activity
  - 🔴 **Red** — Anomaly detected (fight, explosion, assault, etc.)
- Full **web interface** — upload any video, view results in browser

---

## How It Works

Normal Surveillance Videos
↓
Frame Extraction (OpenCV)
↓
ConvLSTM Autoencoder Training (PyTorch)
→ Learns to reconstruct normal frames
↓
Saved Model Checkpoint (.pt)
↓
New Video Uploaded via Web UI
↓
FastAPI Backend → Inference Pipeline
→ Per-frame reconstruction error
→ Adaptive thresholding (absolute + relative z-score)
↓
Red/Yellow/Green Heatmap Overlay (OpenCV)
↓
Result displayed in browser with anomaly timeline

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | PyTorch — ConvLSTM Autoencoder |
| Video Processing | OpenCV, FFmpeg |
| Backend API | FastAPI + BackgroundTasks |
| Frontend | Vanilla HTML/CSS/JS (dark glassmorphism UI) |
| Anomaly Scoring | Absolute threshold + adaptive per-video z-score |

---

## Project Structure
```
Video-Anomaly-Detection/
├── models/
│   ├── convlstm.py          # ConvLSTM cell + multi-layer wrapper
│   ├── autoencoder.py       # Spatial encoder/decoder + ConvLSTM AE
│   └── dataset.py           # PyTorch dataset (PNG frame sequences)
├── training/
│   ├── config.py            # All hyperparameters and paths
│   ├── train.py             # Training loop with checkpoint resuming
│   └── evaluate.py          # AUC-ROC, precision, recall, F1
├── inference/
│   ├── detect.py            # Anomaly detection + adaptive thresholding
│   └── heatmap.py           # Red/yellow/green heatmap overlay generator
├── backend/
│   ├── app.py               # FastAPI app with CORS + static file serving
│   ├── routes.py            # Upload, status, result API endpoints
│   └── utils.py             # Video validation + anomaly report generation
├── frontend/
│   ├── index.html           # Main web UI
│   ├── style.css            # Dark glassmorphism design
│   └── app.js               # Upload, polling, result rendering
├── scripts/
│   ├── extract_dataset_frames.py   # Pre-extract PNG frames from videos
│   ├── recompute_thresholds.py     # Recompute detection thresholds
│   ├── fix_split.py                # Fix train/val dataset split
│   └── run_full_inference.py       # Run detection + heatmap on any video
├── data/
│   ├── train/               # Raw normal training videos (MP4)
│   ├── val/                 # Raw normal validation videos (MP4)
│   ├── test/                # Test videos (normal + anomalous)
│   ├── train_frames/        # Pre-extracted PNG frames (train)
│   └── val_frames/          # Pre-extracted PNG frames (val)
├── checkpoints/             # Saved model weights (.pt) + thresholds (.npy)
├── outputs/                 # Processed heatmap videos
└── uploads/                 # Temporarily uploaded videos (auto-cleaned)
```
---

## Setup

### 1. Clone and create environment

```bash
git clone <your-repo-url>
cd Video-Anomaly-Detection
conda create -n anomaly-detection python=3.11 -y
conda activate anomaly-detection
pip install -r requirements.txt
```

### 2. Prepare dataset

Download the **UCF-Crime** dataset from: https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0

Download:
- `Training-Normal-Videos-Part1.zip` → extract into `data/train/`
- `Testing_Normal_Videos.zip` → extract into `data/val/`

Then fix the train/val split:
```bash
python scripts/fix_split.py
```

Then pre-extract frames:
```bash
python scripts/extract_dataset_frames.py
```

### 3. Train the model

```bash
python training/train.py
```

Training resumes automatically from the last checkpoint if interrupted.
Checkpoint saved to `checkpoints/best_model.pt`.
Loss curve saved to `outputs/loss_curve.png`.

### 4. Recompute thresholds (optional, after training)

```bash
python scripts/recompute_thresholds.py
```

### 5. Run inference on a video (command line)

```bash
python scripts/run_full_inference.py data/test/your_video.mp4
```

Output: `outputs/your_video_heatmap.mp4`

### 6. Start the web application

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Open in browser: `http://localhost:8000/static/index.html`

---

## Model Architecture

**ConvLSTM Autoencoder**

Input: sequence of N grayscale frames (N × 1 × 48 × 48)
↓
SpatialEncoder: Conv2D × 3 (16 → 32 → 64 channels, stride 2)
↓ (each frame encoded to 64 × 6 × 6)
ConvLSTM: 2-layer ConvLSTM (hidden=32, kernel=3×3)
↓
Bottleneck: Conv2D (32 → 64 channels)
↓
SpatialDecoder: ConvTranspose2D × 3 (64 → 32 → 16 → 1)
↓
Output: reconstructed sequence (N × 1 × 48 × 48)
Anomaly score = MSE(output, input) per frame

---

## Anomaly Detection Logic

Two-signal adaptive thresholding:

For each frame:
absolute_flag = score >= threshold_low (from val set percentile)
relative_z    = (score - video_baseline) / video_std
if score >= threshold_high OR relative_z >= 3.0 → ANOMALY (RED)
if score >= threshold_low  OR relative_z >= 1.75 → SUSPICIOUS (YELLOW)
else → NORMAL (GREEN)

This catches anomalies even in videos where absolute error scale is small,
by flagging frames that deviate significantly from that video's own baseline.

---

## Dataset

| Dataset | Used for | Source |
|---|---|---|
| UCF-Crime (Normal Videos) | Training + Validation | https://www.crcv.ucf.edu/projects/real-world/ |
| UCF-Crime (Anomaly Videos) | Testing | Same link above |
| XD-Violence | Testing (additional) | https://roc-ng.github.io/XD-Violence/ |

UCF-Crime contains 13 real anomaly categories:
Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting,
Road Accident, Robbery, Shooting, Shoplifting, Stealing, Vandalism

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload` | Upload video, returns `job_id` |
| GET | `/api/status/{job_id}` | Poll processing status |
| GET | `/api/result/{job_id}` | Get heatmap video URL + anomaly report |
| GET | `/api/jobs` | List all processed jobs |

---

## Known Limitations

- Trained at 48×48 resolution — adequate for proof of concept, retrain at higher resolution for production
- CPU inference is slow (~1-2 min per video) — GPU deployment recommended for real-time use
- Model trained on indoor/outdoor CCTV footage — satellite footage requires separate training data
- Unsupervised approach flags *any* deviation from training distribution — not a classifier of specific crime types

---

## Future Work

- [ ] Train at higher resolution (128×128 or 256×256) with GPU
- [ ] Add XD-Violence and satellite imagery datasets
- [ ] Real-time RTSP stream processing
- [ ] Crime type classifier on top of anomaly detector
- [ ] Docker deployment
- [ ] GPU-accelerated inference API