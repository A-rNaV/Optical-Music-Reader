import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from omr.config import PROCESSED_DIR, RAW_DIR
from omr.dataset.tokenizer import Tokenizer


def load_dataset():

    with open(
        PROCESSED_DIR / "dataset.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def analyze_tokens(dataset):

    tokenizer = Tokenizer()

    frequencies = Counter()
    lengths = []

    for sample in dataset:

        with open(
            sample["label"],
            "r",
            encoding="utf-8"
        ) as f:
            text = f.read()

        tokens = tokenizer.tokenize(text)

        frequencies.update(tokens)

        lengths.append(len(tokens))

    return frequencies, lengths


def analyze_images(dataset):

    widths = []
    heights = []
    aspect_ratios = []

    for sample in dataset:

        image_path = Path(sample["image"])

        with Image.open(image_path) as img:

            width, height = img.size

        widths.append(width)
        heights.append(height)
        aspect_ratios.append(width / height)

    return widths, heights, aspect_ratios


def percentile(values, p):

    return np.percentile(values, p)


def print_frequency_stats(frequencies):

    print("=" * 60)
    print("TOKEN FREQUENCY STATISTICS")
    print("=" * 60)

    total_tokens = sum(frequencies.values())
    unique_tokens = len(frequencies)

    print(f"Total Tokens       : {total_tokens}")
    print(f"Unique Tokens      : {unique_tokens}")
    print(
        f"Tokens appearing once : "
        f"{sum(v == 1 for v in frequencies.values())}"
    )
    print(
        f"Tokens appearing <=5  : "
        f"{sum(v <= 5 for v in frequencies.values())}"
    )
    print(
        f"Tokens appearing <=10 : "
        f"{sum(v <= 10 for v in frequencies.values())}"
    )

    print()

    print("TOP 30 MOST FREQUENT TOKENS")
    print("-" * 60)

    for token, count in frequencies.most_common(30):

        print(f"{token:<30} {count}")

    print()


def print_sequence_stats(lengths):

    print("=" * 60)
    print("SEQUENCE LENGTH STATISTICS")
    print("=" * 60)

    print(f"Minimum       : {min(lengths)}")
    print(f"Maximum       : {max(lengths)}")
    print(f"Mean          : {np.mean(lengths):.2f}")
    print(f"Median        : {np.median(lengths):.2f}")
    print(f"90th percentile: {percentile(lengths, 90):.2f}")
    print(f"95th percentile: {percentile(lengths, 95):.2f}")
    print(f"99th percentile: {percentile(lengths, 99):.2f}")
    print(f"> 436 tokens : {sum(x > 436 for x in lengths)}")
    print(f"> 512 tokens : {sum(x > 512 for x in lengths)}")
    print(f"> 612 tokens : {sum(x > 612 for x in lengths)}")
    print(f"> 768 tokens : {sum(x > 768 for x in lengths)}")

    print()


def print_image_stats(widths, heights, aspect_ratios):

    print("=" * 60)
    print("IMAGE DIMENSION STATISTICS")
    print("=" * 60)

    print("WIDTH")
    print(f"Minimum        : {min(widths)}")
    print(f"Maximum        : {max(widths)}")
    print(f"Mean           : {np.mean(widths):.2f}")
    print(f"Median         : {np.median(widths):.2f}")
    print(f"95th percentile: {percentile(widths, 95):.2f}")

    print()

    print("HEIGHT")
    print(f"Minimum        : {min(heights)}")
    print(f"Maximum        : {max(heights)}")
    print(f"Mean           : {np.mean(heights):.2f}")
    print(f"Median         : {np.median(heights):.2f}")
    print(f"95th percentile: {percentile(heights, 95):.2f}")

    print()

    print("ASPECT RATIO (WIDTH / HEIGHT)")
    print(f"Minimum        : {min(aspect_ratios):.2f}")
    print(f"Maximum        : {max(aspect_ratios):.2f}")
    print(f"Mean           : {np.mean(aspect_ratios):.2f}")
    print(f"Median         : {np.median(aspect_ratios):.2f}")
    print(
        f"95th percentile: "
        f"{percentile(aspect_ratios, 95):.2f}"
    )

    print()


if __name__ == "__main__":

    dataset = load_dataset()

    print(f"Total Samples: {len(dataset)}")
    print()

    frequencies, lengths = analyze_tokens(dataset)

    widths, heights, aspect_ratios = analyze_images(dataset)

    print_frequency_stats(frequencies)

    print_sequence_stats(lengths)

    print_image_stats(
        widths,
        heights,
        aspect_ratios
    )