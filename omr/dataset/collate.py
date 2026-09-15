import torch
import torch.nn.functional as F
def collate_fn(batch):
    max_width=max(sample["image"].shape[2] for sample in batch)
    images=[]
    for sample in batch:
        image=sample["image"]
        padding=max_width-image.shape[2]
        image=F.pad(image, (0, padding, 0, 0), value=0)
        images.append(image)
    images=torch.stack(images)
    labels=torch.stack([sample["labels"] for sample in batch])
    lengths=torch.tensor([sample["length"] for sample in batch])
    img_pths=[sample["image_path"] for sample in batch]
    return {
        "images":images,
        "labels":labels,
        "lengths":lengths,
        "image_paths":img_pths
    }