import os
import sys
import cv2
import argparse
from tqdm import tqdm


def extract_frames(video_path, output_dir, fps=10, image_size=128, grayscale=True):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps == 0:
        print("ERROR: Could not read video FPS.")
        return

    # Save every Nth frame to match target FPS
    frame_interval = max(1, int(video_fps / fps))
    frame_count = 0
    saved_count = 0

    print(f"Video FPS: {video_fps:.1f} | Extracting at: {fps} FPS | Interval: every {frame_interval} frames")

    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                if grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (image_size, image_size))
                filename = os.path.join(output_dir, f"frame_{saved_count:06d}.png")
                cv2.imwrite(filename, frame)
                saved_count += 1

            frame_count += 1
            pbar.update(1)

    cap.release()
    print(f"Done. Saved {saved_count} frames to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video_path",  type=str, help="Path to input video")
    parser.add_argument("output_dir",  type=str, help="Directory to save extracted frames")
    parser.add_argument("--fps",       type=int, default=10,  help="Target FPS to extract at")
    parser.add_argument("--size",      type=int, default=128, help="Resize frames to this size (square)")
    parser.add_argument("--grayscale", action="store_true",   help="Save as grayscale")
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir, args.fps, args.size, args.grayscale)