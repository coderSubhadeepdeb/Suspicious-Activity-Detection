import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoFrameDataset(Dataset):
    """
    Loads sequences of pre-extracted grayscale frames from disk.

    Augmentation (applied only when augment=True, i.e. during training):
    - Random horizontal flip  (same decision applied to all frames in sequence)
    - Random brightness / contrast jitter   (same factor → temporal consistency)
    - Random Gaussian noise               (independent per frame is fine)
    - Random temporal reversal            (plays sequence backwards, rare)

    Stride is now 3 (was 5) → ~67 % more training sequences per video.
    """

    def __init__(self, data_dir, sequence_length=10, image_size=64,
                 stride=3, augment=False):
        self.sequence_length = sequence_length
        self.image_size      = image_size
        self.augment         = augment
        self.sequences       = []

        folders = sorted([
            os.path.join(data_dir, d) for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        for folder in folders:
            frames = sorted([
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(".png")
            ])
            if len(frames) < sequence_length:
                continue
            for i in range(0, len(frames) - sequence_length + 1, stride):
                self.sequences.append(frames[i:i + sequence_length])

        print(f"[Dataset] {'Train' if augment else 'Val/Test'} — "
              f"{len(self.sequences)} sequences  (stride={stride}, "
              f"aug={augment})  from {data_dir}")

    def __len__(self):
        return len(self.sequences)

    def _load_frame(self, path):
        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            frame = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        return frame.astype(np.float32) / 255.0

    def __getitem__(self, idx):
        frame_paths = self.sequences[idx]
        frames = [self._load_frame(p) for p in frame_paths]

        # ── Augmentation (training only) ─────────────────────────────────────
        if self.augment:
            # 1. Horizontal flip (same for all frames in sequence)
            if random.random() < 0.5:
                frames = [np.fliplr(f).copy() for f in frames]

            # 2. Brightness / contrast jitter (same factor for whole clip)
            if random.random() < 0.6:
                alpha = random.uniform(0.75, 1.25)   # contrast
                beta  = random.uniform(-0.08, 0.08)  # brightness
                frames = [np.clip(f * alpha + beta, 0.0, 1.0) for f in frames]

            # 3. Gaussian noise (independent per frame — adds texture variation)
            if random.random() < 0.4:
                sigma = random.uniform(0.005, 0.025)
                frames = [
                    np.clip(f + np.random.normal(0, sigma, f.shape).astype(np.float32),
                            0.0, 1.0)
                    for f in frames
                ]

            # 4. Temporal reversal (rare — network must learn both directions)
            if random.random() < 0.2:
                frames = list(reversed(frames))

        # ── Stack → (seq_len, 1, H, W) tensor ───────────────────────────────
        frames_np = np.stack(frames, axis=0)           # (T, H, W)
        frames_np = np.expand_dims(frames_np, axis=1)  # (T, 1, H, W)
        return torch.tensor(frames_np, dtype=torch.float32)