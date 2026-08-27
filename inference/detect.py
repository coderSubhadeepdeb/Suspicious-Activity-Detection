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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    cap         = cv2.VideoCapture(video_path)
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


def smooth_scores(scores, window=3):
    """
    Apply a causal moving-average over the raw per-frame anomaly scores.
    Reduces flickering caused by single-frame reconstruction noise.
    The window is causal (only looks back) so there is no lookahead bias.
    """
    smoothed = []
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        smoothed.append(float(np.mean(scores[start:i + 1])))
    return smoothed


def classify_frame(score, threshold_low, threshold_high,
                   video_mean, video_std):
    """
    Two-tier decision:
    1. Absolute check against trained thresholds (primary, model-calibrated).
    2. Relative check against THIS video's own distribution.
    Combines both absolute and relative statuses, returning the highest risk level.
    """
    # Relative z-score within this video
    z = (score - video_mean) / (video_std + 1e-8)

    # Absolute status
    if score >= threshold_high:
        abs_status = "ANOMALY"
    elif score >= threshold_low:
        abs_status = "SUSPICIOUS"
    else:
        abs_status = "NORMAL"

    # Relative status
    if z >= 2.0:
        rel_status = "ANOMALY"
    elif z >= 1.2:
        rel_status = "SUSPICIOUS"
    else:
        rel_status = "NORMAL"

    # Return the maximum threat level of the two (upgrade policy)
    status_order = {"NORMAL": 0, "SUSPICIOUS": 1, "ANOMALY": 2}
    if status_order[rel_status] > status_order[abs_status]:
        return rel_status
    return abs_status


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

    # ── Raw per-frame scores ──────────────────────────────────────────────────
    raw_scores  = []
    raw_maps    = []

    with torch.no_grad():
        for i in range(total_frames - SEQUENCE_LENGTH + 1):
            seq = frames_gray[i:i + SEQUENCE_LENGTH]
            seq_tensor = np.stack(seq, axis=0)                  # (T, H, W)
            seq_tensor = np.expand_dims(seq_tensor, axis=1)     # (T, 1, H, W)
            seq_tensor = torch.tensor(seq_tensor, dtype=torch.float32).unsqueeze(0).to(device)

            output = model(seq_tensor)

            # Per-pixel squared error for the LAST frame in the sequence
            # dim=2 → across channel (only 1 grayscale channel)
            error_map = ((output - seq_tensor) ** 2).squeeze(0)  # (T, 1, H, W)
            last_err  = error_map[-1, 0].cpu().numpy()            # (H, W)

            raw_scores.append(float(np.mean(last_err)))
            raw_maps.append(last_err)

    # ── Temporal smoothing ────────────────────────────────────────────────────
    smoothed = smooth_scores(raw_scores, window=3)

    # ── Video-level statistics (for relative check) ───────────────────────────
    scores_arr  = np.array(smoothed)
    video_mean  = float(np.mean(scores_arr))
    video_std   = float(np.std(scores_arr))

    print(f"Video stats — mean: {video_mean:.6f} | std: {video_std:.6f} | "
          f"max: {scores_arr.max():.6f}")

    # ── Classify each frame ───────────────────────────────────────────────────
    results = []
    for i, (raw_score, smooth_score, err_map) in enumerate(
            zip(raw_scores, smoothed, raw_maps)):
        frame_idx = i + SEQUENCE_LENGTH - 1
        status    = classify_frame(smooth_score, threshold_low, threshold_high,
                                   video_mean, video_std)
        results.append({
            'frame_idx':     frame_idx,
            'anomaly_score': smooth_score,   # smoothed score for classification
            'raw_score':     raw_score,      # original score for logging
            'error_map':     err_map,
            'original_frame': frames_orig[frame_idx],
            'status':        status
        })

    print(f"Detection complete. Processed {len(results)} frames.")
    counts = {s: sum(1 for r in results if r['status'] == s)
              for s in ('NORMAL', 'SUSPICIOUS', 'ANOMALY')}
    print(f"  NORMAL: {counts['NORMAL']} | "
          f"SUSPICIOUS: {counts['SUSPICIOUS']} | "
          f"ANOMALY: {counts['ANOMALY']}")

    return results, threshold_low, threshold_high


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detect.py <video_path>")
    else:
        results, tl, th = detect_anomalies(sys.argv[1])
        for r in results:
            print(f"Frame {r['frame_idx']:4d} | "
                  f"Score: {r['anomaly_score']:.6f} | "
                  f"Raw: {r['raw_score']:.6f} | "
                  f"{r['status']}")