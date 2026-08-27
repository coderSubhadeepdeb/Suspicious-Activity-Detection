import os
import cv2
from tqdm import tqdm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_frames_from_videos(video_dir, output_dir, image_size=64, target_fps=5):
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for vf in tqdm(video_files, desc=f"Extracting {video_dir}"):
        video_path = os.path.join(video_dir, vf)
        video_name = os.path.splitext(vf)[0]
        out_folder = os.path.join(output_dir, video_name)

        if os.path.exists(out_folder) and len(os.listdir(out_folder)) > 0:
            continue

        os.makedirs(out_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_every = max(1, int(fps / target_fps)) if fps > 0 else 1

        count = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (image_size, image_size))
                cv2.imwrite(os.path.join(out_folder, f"frame_{saved:05d}.png"), gray)
                saved += 1
            count += 1
        cap.release()


if __name__ == "__main__":
    # image_size MUST match IMAGE_SIZE in training/config.py (currently 64)
    extract_frames_from_videos(
        os.path.join(BASE, "data", "train"),
        os.path.join(BASE, "data", "train_frames"),
        image_size=64, target_fps=5
    )
    extract_frames_from_videos(
        os.path.join(BASE, "data", "val"),
        os.path.join(BASE, "data", "val_frames"),
        image_size=64, target_fps=5
    )
    print("Frame extraction complete!")