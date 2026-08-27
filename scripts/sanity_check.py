import sys
sys.path.insert(0, '.')
import torch
from models.autoencoder import ConvLSTMAE
from training.config import ENCODER_CHANNELS, CONVLSTM_HIDDEN, SEQUENCE_LENGTH, IMAGE_SIZE

print(f"Config: IMAGE_SIZE={IMAGE_SIZE}, SEQ_LEN={SEQUENCE_LENGTH}")
print(f"  ENCODER_CHANNELS={ENCODER_CHANNELS}")
print(f"  CONVLSTM_HIDDEN={CONVLSTM_HIDDEN}")

model = ConvLSTMAE(ENCODER_CHANNELS, CONVLSTM_HIDDEN, SEQUENCE_LENGTH)
total = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total:,}")

# Test with batch_size=2
x = torch.randn(2, SEQUENCE_LENGTH, 1, IMAGE_SIZE, IMAGE_SIZE)
print(f"  Input shape:  {list(x.shape)}")
with torch.no_grad():
    out = model(x)
print(f"  Output shape: {list(out.shape)}")
assert out.shape == x.shape, f"SHAPE MISMATCH: {out.shape} vs {x.shape}"
print("\n  [PASS] Architecture is valid!")
print("  Ready to retrain from scratch.")
