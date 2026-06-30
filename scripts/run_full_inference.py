import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.detect import detect_anomalies
from inference.heatmap import generate_heatmap_video
from training.config import OUTPUT_DIR

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_full_inference.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    results, threshold_low, threshold_high = detect_anomalies(video_path)

    output_filename = os.path.splitext(os.path.basename(video_path))[0] + "_heatmap.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    generate_heatmap_video(results, threshold_low, threshold_high, output_path)
    print(f"\nDone! Heatmap video saved at: {output_path}")