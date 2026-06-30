import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train_frames")
VAL_DIR = os.path.join(DATA_DIR, "val_frames")
TEST_DIR = os.path.join(DATA_DIR, "test")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

SEQUENCE_LENGTH = 6
IMAGE_SIZE = 48
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 25
ENCODER_CHANNELS = [16, 32, 64]
CONVLSTM_HIDDEN = 32

THRESHOLD_LOW = 0.02
THRESHOLD_HIGH = 0.05

NUM_WORKERS = 4
PIN_MEMORY = False
SAVE_EVERY = 1