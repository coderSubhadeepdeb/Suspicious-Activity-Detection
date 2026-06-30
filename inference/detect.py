import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import numpy as np

from models.autoencoder import ConvLSTMAE
from training.config import *


def load_model(checkpoint_path, device):
    model = ConvLSTMAE(ENCODER_CHANNELS, CONVLSTM_HIDDEN, SEQUENCE_LENGTH).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_thresholds():
    threshold_path = os.path.join(CHECKPOINT_DIR, 'thresholds.npy')
    if os.path.exists(threshold_path):
        thresholds = np.load(threshold_path)
        return float(thresholds[0]), float(thresholds[1])
    return THRESHOLD_LOW, THRESHOLD_HIGH


def extract_frames_from_video(video_path, image_size):
    cap = cv2.VideoCapture(video_path)
    frames_gray = []
    frames_orig = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_orig.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (image_size, image_size))
        gray = gray.astype(np.float32) / 255.0
        frames_gray.append(gray)

    cap.release()
    return frames_gray, frames_orig


def detect_anomalies(video_path, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    model = load_model(checkpoint_path, device)
    threshold_low, threshold_high = load_thresholds()
    print(f"Thresholds — LOW: {threshold_low:.6f}, HIGH: {threshold_high:.6f}")

    frames_gray, frames_orig = extract_frames_from_video(video_path, IMAGE_SIZE)
    total_frames = len(frames_gray)
    print(f"Total frames extracted: {total_frames}")

    if total_frames < SEQUENCE_LENGTH:
        raise ValueError(f"Video too short. Need at least {SEQUENCE_LENGTH} frames.")

    raw_results = []

    with torch.no_grad():
        for i in range(total_frames - SEQUENCE_LENGTH + 1):
            seq = frames_gray[i:i + SEQUENCE_LENGTH]
            seq_tensor = np.stack(seq, axis=0)
            seq_tensor = np.expand_dims(seq_tensor, axis=1)
            seq_tensor = torch.tensor(seq_tensor, dtype=torch.float32).unsqueeze(0).to(device)

            output = model(seq_tensor)

            # Per-pixel error for the last frame in the sequence
            error_map = torch.mean((output - seq_tensor) ** 2, dim=2)
            last_error_map = error_map[0, -1].cpu().numpy()

            # Overall score for this frame
            anomaly_score = float(np.mean(last_error_map))

            raw_results.append({
                'frame_idx': i + SEQUENCE_LENGTH - 1,
                'anomaly_score': anomaly_score,
                'error_map': last_error_map,
                'original_frame': frames_orig[i + SEQUENCE_LENGTH - 1]
            })

    # --- Adaptive relative scoring on top of absolute thresholds ---
    scores = np.array([r['anomaly_score'] for r in raw_results])
    baseline = np.percentile(scores, 25)  # bottom 25% = this video's "calm" baseline
    spread = np.std(scores) + 1e-8

    results = []
    for r in raw_results:
        score = r['anomaly_score']
        relative_z = (score - baseline) / spread

        # Combine absolute thresholds with relative (within-video) deviation
        if score >= threshold_high or relative_z >= 3.0:
            status = "ANOMALY"
        elif score >= threshold_low or relative_z >= 1.75:
            status = "SUSPICIOUS"
        else:
            status = "NORMAL"

        r['status'] = status
        results.append(r)

    print(f"Detection complete. Processed {len(results)} frames.")
    print(f"Video baseline: {baseline:.6f} | spread (std): {spread:.6f}")
    return results, threshold_low, threshold_high


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detect.py <video_path>")
    else:
        results, tl, th = detect_anomalies(sys.argv[1])
        for r in results:
            print(f"Frame {r['frame_idx']:4d} | Score: {r['anomaly_score']:.6f} | {r['status']}")