import os
import sys
import urllib.request
import zipfile
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

UCSD_PED2_URL = "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"


class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def extract_frames_from_video(video_path, output_dir, image_size=128):
    import cv2
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (image_size, image_size))
        cv2.imwrite(os.path.join(output_dir, f"frame_{count:06d}.png"), gray)
        count += 1
    cap.release()
    return count


def organize_ucsd_ped2(extracted_dir):
    """Organize UCSD Ped2 into data/train, data/val, data/test"""
    import shutil

    ped2_train = os.path.join(extracted_dir, "UCSD_Anomaly_Dataset.v1p2", "UCSDped2", "Train")
    ped2_test  = os.path.join(extracted_dir, "UCSD_Anomaly_Dataset.v1p2", "UCSDped2", "Test")

    train_out = os.path.join(DATA_DIR, "train")
    val_out   = os.path.join(DATA_DIR, "val")
    test_out  = os.path.join(DATA_DIR, "test")

    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out,   exist_ok=True)
    os.makedirs(test_out,  exist_ok=True)

    if not os.path.exists(ped2_train):
        print(f"Could not find UCSD Ped2 train folder at {ped2_train}")
        print("Please organize your dataset manually into data/train, data/val, data/test")
        return

    # Training folders (normal only) — use first 13 for train, last 3 for val
    train_clips = sorted([
        d for d in os.listdir(ped2_train)
        if os.path.isdir(os.path.join(ped2_train, d))
    ])

    split = int(len(train_clips) * 0.8)
    train_set = train_clips[:split]
    val_set   = train_clips[split:]

    print(f"Organizing {len(train_set)} clips into train, {len(val_set)} into val...")

    for clip in train_set:
        src = os.path.join(ped2_train, clip)
        dst = os.path.join(train_out, clip)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)

    for clip in val_set:
        src = os.path.join(ped2_train, clip)
        dst = os.path.join(val_out, clip)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)

    # Test set
    if os.path.exists(ped2_test):
        test_clips = sorted([
            d for d in os.listdir(ped2_test)
            if os.path.isdir(os.path.join(ped2_test, d))
        ])
        for clip in test_clips:
            src = os.path.join(ped2_test, clip)
            dst = os.path.join(test_out, clip)
            if not os.path.exists(dst):
                shutil.copytree(src, dst)

    print("Dataset organized successfully.")
    print(f"  Train : {train_out}")
    print(f"  Val   : {val_out}")
    print(f"  Test  : {test_out}")


def manual_instructions():
    print("\n" + "="*60)
    print("MANUAL DATASET SETUP — UCSD Ped2")
    print("="*60)
    print("1. Go to: http://www.svcl.ucsd.edu/projects/anomaly/dataset.html")
    print("2. Download 'UCSD_Anomaly_Dataset.tar.gz'")
    print("3. Extract it")
    print("4. Copy the Train folders (normal clips, .tif frames) into:")
    print(f"     {os.path.join(DATA_DIR, 'train')}")
    print("5. Copy some Train folders (held out) into:")
    print(f"     {os.path.join(DATA_DIR, 'val')}")
    print("6. Copy the Test folders into:")
    print(f"     {os.path.join(DATA_DIR, 'test')}")
    print("\nEach subfolder should contain .tif or .png frame images.")
    print("="*60 + "\n")


if __name__ == "__main__":
    manual_instructions()