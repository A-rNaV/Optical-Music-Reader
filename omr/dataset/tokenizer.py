import json
class Tokenizer:
    def __init__(self, vocab_path=None):
        self.token_to_id={}
        self.id_to_token={}
        if vocab_path is not None:
            self.load_vocab(vocab_path)

    def tokenize(self, text):
        tokens=["<BOS>"]
        lines=text.splitlines()
        for line in lines:
            columns=line.split("\t")
            for i, col in enumerate(columns):
                symbols=col.split()
                for sidx, symbol in enumerate(symbols):
                    tokens.append(symbol)
                    if sidx!=len(symbols)-1:
                        tokens.append("<SPACE>")
                if i!=len(columns)-1:
                    tokens.append("<TAB>")
            tokens.append("<NEWLINE>")
        tokens.append("<EOS>")
        return tokens

    def detokenize(self, tokens):
        output=[]
        current_line=[]
        for token in tokens:
            if token in ("<BOS>", "<EOS>", "<PAD>"):
                continue
            elif token=="<TAB>":
                current_line.append("\t")
            elif token=="<SPACE>":
                current_line.append(" ")
            elif token=="<NEWLINE>":
                output.append("".join(current_line))
                current_line=[]
            else:
                current_line.append(token)
        return "\n".join(output)

    def encode(self, tokens):
        ids=[]
        unk=self.token_to_id["<UNK>"]
        for token in tokens:
            ids.append(self.token_to_id.get(token, unk))
        return ids

    def decode(self, ids):
        tokens=[]
        for idx in ids:
            tokens.append(self.id_to_token[str(idx)])
        return tokens

    def load_vocab(self, vocab_path):
        with open(vocab_path, "r") as f:
            vocab=json.load(f)
        self.token_to_id=vocab["token_to_id"]
        self.id_to_token=vocab["id_to_token"]
