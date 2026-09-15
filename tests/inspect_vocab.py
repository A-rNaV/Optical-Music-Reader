import json

from omr.config import PROCESSED_DIR


def load_vocab():

    with open(PROCESSED_DIR / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    return vocab


def basic_statistics(vocab):

    token_to_id = vocab["token_to_id"]

    print("=" * 60)
    print("VOCABULARY STATISTICS")
    print("=" * 60)

    print(f"Vocabulary Size      : {vocab['vocab_size']}")
    print(f"Special Tokens       : 6")
    print(f"Regular Tokens       : {vocab['vocab_size'] - 6}")

    print()


def inspect_tokens(vocab):

    token_to_id = vocab["token_to_id"]

    tokens = list(token_to_id.keys())

    print("=" * 60)
    print("FIRST 50 TOKENS")
    print("=" * 60)

    for token in tokens[:50]:
        print(token)

    print()

    print("=" * 60)
    print("LAST 50 TOKENS")
    print("=" * 60)

    for token in tokens[-50:]:
        print(token)

    print()


def token_lengths(vocab):

    token_to_id = vocab["token_to_id"]

    tokens = list(token_to_id.keys())

    longest = max(tokens, key=len)
    shortest = min(tokens, key=len)

    print("=" * 60)
    print("TOKEN LENGTHS")
    print("=" * 60)

    print(f"Longest Token  : {longest}")
    print(f"Length         : {len(longest)}")

    print()

    print(f"Shortest Token : {shortest}")
    print(f"Length         : {len(shortest)}")

    print()


def prefix_statistics(vocab):

    token_to_id = vocab["token_to_id"]

    tokens = list(token_to_id.keys())

    clefs = 0
    key_sig = 0
    time_sig = 0
    bars = 0
    notes = 0

    for token in tokens:

        if token.startswith("*clef"):
            clefs += 1

        elif token.startswith("*k"):
            key_sig += 1

        elif token.startswith("*M"):
            time_sig += 1

        elif token.startswith("="):
            bars += 1

        else:
            notes += 1

    print("=" * 60)
    print("TOKEN CATEGORIES")
    print("=" * 60)

    print(f"Clefs           : {clefs}")
    print(f"Key Signatures  : {key_sig}")
    print(f"Time Signatures : {time_sig}")
    print(f"Barlines        : {bars}")
    print(f"Other Tokens    : {notes}")

    print()


if __name__ == "__main__":

    vocab = load_vocab()

    basic_statistics(vocab)

    inspect_tokens(vocab)

    token_lengths(vocab)

    prefix_statistics(vocab)