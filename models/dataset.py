import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoFrameDataset(Dataset):
    def __init__(self, data_dir, sequence_length=6, image_size=48, stride=5):
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.sequences = []

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

        print(f"Total sequences ready: {len(self.sequences)} from {data_dir}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frame_paths = self.sequences[idx]
        frames = []
        for path in frame_paths:
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                frame = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        frames = np.stack(frames, axis=0)
        frames = np.expand_dims(frames, axis=1)
        return torch.tensor(frames, dtype=torch.float32)