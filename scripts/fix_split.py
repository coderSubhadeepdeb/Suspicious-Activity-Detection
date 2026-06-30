import os
import shutil

val_dir   = r'E:\Video-Anomaly-detection\data\val'
train_dir = r'E:\Video-Anomaly-detection\data\train'

files = sorted([f for f in os.listdir(val_dir) if f.lower().endswith('.mp4')])
to_move = files[:120]

for f in to_move:
    shutil.move(os.path.join(val_dir, f), os.path.join(train_dir, f))

print(f'Moved {len(to_move)} files to train')
print(f'Train now has: {len(os.listdir(train_dir))} files')
print(f'Val now has:   {len(os.listdir(val_dir))} files')