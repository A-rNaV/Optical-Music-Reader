from collections import OrderedDict
import json
from omr.config import PROCESSED_DIR
from omr.dataset.tokenizer import Tokenizer

SPECIAL_TOKENS = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
    "<SPACE>",
    "<TAB>",
    "<NEWLINE>"
]

class VocabBuilder:
    def __init__(self):
        self.vocab=OrderedDict()
        for token in SPECIAL_TOKENS:
            self.vocab[token]=len(self.vocab)
        self.tokenizer=Tokenizer()

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
    
    def build(self):
        with open(PROCESSED_DIR/"dataset.json", "r", encoding="utf-8") as f:
            dataset=json.load(f)
            for sample in dataset:
                with open(sample["label"], "r", encoding="utf-8") as f:
                    text=f.read()
                tokens=self.tokenizer.tokenize(text)
                self.add_tokens(tokens)

    def save(self):
        token_to_id = dict(self.vocab)
        id_to_token = {
            str(v): k
            for k, v in self.vocab.items()
        }
        output = {
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "vocab_size": len(self.vocab)
        }
        save_path = PROCESSED_DIR / "vocab.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)


if __name__ == "__main__":
    builder = VocabBuilder()
    builder.build()
    builder.save()
    print(f"Vocabulary Size : {len(builder.vocab)}")