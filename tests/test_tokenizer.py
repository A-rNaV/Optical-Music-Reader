import json
from omr.config import PROCESSED_DIR
from omr.dataset.tokenizer import Tokenizer
with open(PROCESSED_DIR / "dataset.json") as f:
    dataset = json.load(f)
tokenizer = Tokenizer()
for sample in dataset[:100]:
    with open(sample["label"], "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.tokenize(text)
    reconstructed = tokenizer.detokenize(tokens)
    assert text.strip() == reconstructed.strip(), \
        f"Tokenizer failed for {sample['label']}"
print("All tokenizer tests passed!")