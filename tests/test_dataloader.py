from torch.utils.data import DataLoader

from omr.dataset.dataset import OMRDataset
from omr.dataset.collate import collate_fn


dataset = OMRDataset(max_length=512)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

batch = next(iter(loader))

print("Images shape :", batch["images"].shape)
print("Labels shape :", batch["labels"].shape)
print("Lengths      :", batch["lengths"])
print("Image paths  :", len(batch["image_paths"]))

assert batch["images"].shape[0] == 4
assert batch["images"].shape[1] == 1
assert batch["images"].shape[2] == 256
assert batch["labels"].shape == (4, 512)

print()
print("DataLoader test passed!")