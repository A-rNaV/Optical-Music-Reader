from pathlib import Path
import json
from omr.config import RAW_DIR, PROCESSED_DIR

class DatasetBuilder:
    def __init__(self, root_dir):
        self.root_dir=Path(root_dir)
        self.samples=[]

    def build(self):
        print(self.root_dir)
        print(self.root_dir.resolve())
        images=self.root_dir.rglob("original_*.jpg")
        for img_path in images:
            if "_distorted" in img_path.name:
                continue

            label_path=img_path.with_suffix(".bekrn")
            if not label_path.exists():
                print(f"Missing label: {img_path}")
                continue

            self.samples.append({
                "image": str(img_path),
                "label": str(label_path)
            })

        print(f"Total Samples: {len(self.samples)}")
        return self.samples

    def save(self, save_path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.samples, f, indent=4)

if __name__=="__main__":
    builder = DatasetBuilder(RAW_DIR/"grandstaff")
    samples = builder.build()
    builder.save(PROCESSED_DIR/"dataset.json")
