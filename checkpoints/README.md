# Model Checkpoints

This directory stores trained model checkpoints for both the Basic CNN (Baseline) and U-Net (Primary Model).

## Checkpoint Files
- `best_basic_cnn.pth`: Model weights with the lowest validation loss for Basic CNN.
- `best_unet.pth`: Model weights with the lowest validation loss for U-Net.
- `latest_basic_cnn.pth`: Latest checkpoint during training for resumption.
- `latest_unet.pth`: Latest checkpoint during training for resumption.

## Loading a Checkpoint in Code

```python
import torch
from models.unet import UNet
from utils.checkpoint import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=2).to(device)

checkpoint = load_checkpoint("checkpoints/best_unet.pth", model, device=device)
print(f"Loaded model from epoch {checkpoint.get('epoch', 0)} with loss {checkpoint.get('val_loss', 0.0):.4f}")
```
