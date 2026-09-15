from omr.dataset.dataset import OMRDataset
dataset=OMRDataset(max_length=512)
print("Dataset size:", len(dataset))
sample = dataset[0]

print()
print("Image shape:", sample["image"].shape)
print("Label shape:", sample["labels"].shape)
print("Sequence length:", sample["length"])
print("Image path:", sample["image_path"])

assert sample["image"].shape[0] == 1
assert sample["image"].shape[1] == 256
assert sample["labels"].shape[0] == 512

print()
print("Dataset test passed!")