import cv2
import os

outputs_dir = "outputs"
test_video = "data/test/explosion2.mp4"

print("Checking test video...")
if os.path.exists(test_video):
    cap = cv2.VideoCapture(test_video)
    print(f"Original Video {test_video}:")
    print(f"  Width : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"  FPS   : {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"  Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    cap.release()
else:
    print(f"Original video not found at {test_video}")

riot_video = "data/test/riot1.mp4"
if os.path.exists(riot_video):
    cap = cv2.VideoCapture(riot_video)
    print(f"Original Video {riot_video}:")
    print(f"  Width : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"  FPS   : {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"  Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    cap.release()
else:
    print(f"Original video not found at {riot_video}")

print("\nChecking generated heatmap videos...")
for f in os.listdir(outputs_dir):
    if f.endswith(".mp4"):
        path = os.path.join(outputs_dir, f)
        cap = cv2.VideoCapture(path)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])
        print(f"Heatmap Video {f} (size={os.path.getsize(path)} bytes):")
        print(f"  Width : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  Codec : {codec}")
        print(f"  FPS   : {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"  Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        cap.release()
