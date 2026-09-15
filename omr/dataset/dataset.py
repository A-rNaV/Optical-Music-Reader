import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from omr.config import RAW_DIR, PROCESSED_DIR
from omr.dataset.tokenizer import Tokenizer

class OMRDataset(Dataset):
    def __init__(self, max_length=512):
        with open(PROCESSED_DIR/"dataset.json", "r", encoding="utf-8") as f:
            self.samples=json.load(f)
            self.tokenizer=Tokenizer(PROCESSED_DIR/"vocab.json")
            self.max_length=max_length
            self.transform=transforms.Compose([transforms.ToTensor()])
            self.pad_id=self.tokenizer.token_to_id["<PAD>"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample=self.samples[index]
        img_pth=sample["image"]
        lbl_pth=sample["label"]
        image=Image.open(img_pth).convert("L")
        image=self.transform(image)
        with open(lbl_pth, "r", encoding="utf-8") as f:
            text=f.read()
        tokens=self.tokenizer.tokenize(text)
        ids=self.tokenizer.encode(tokens)
        original_length=len(ids)
        if len(ids)>self.max_length:
            ids=ids[:self.max_length]
        else:
            ids+=[self.pad_id]*(self.max_length-len(ids))
        labels=torch.tensor(ids, dtype=torch.long)

        return{
            "image":image,
            "labels":labels,
            "length":min(original_length, self.max_length),
            "image_path":img_pth
        }