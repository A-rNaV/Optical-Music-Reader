import torch

from omr.dataset.dataset import OMRDataset
from omr.dataset.collate import collate_fn
from omr.models.cnn_encoder import CNNEncoder
from torch.utils.data import DataLoader


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = OMRDataset(max_length=512)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

batch = next(iter(loader))

images = batch["images"].to(device)

model = CNNEncoder().to(device)

with torch.no_grad():
    features = model(images)

print("Input shape   :", images.shape)
print("Output shape  :", features.shape)
print("Device        :", features.device)

assert features.shape[0] == 4
assert features.shape[1] == 256
assert features.shape[2] == 16

print()
print("CNN test passed!")