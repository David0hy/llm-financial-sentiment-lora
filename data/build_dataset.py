"""Build a labeled financial sentiment dataset for the case study.

Pulls finance-related tweets from the Twitter Financial News Sentiment dataset
(zeroshot/twitter-financial-news-sentiment on HuggingFace), a well-established
public benchmark released under the MIT license.

Three sentiment classes:
  - bearish: negative outlook on a stock/market/economy
  - bullish: positive outlook
  - neutral: factual or non-directional news

The full dataset has ~12k examples. We subsample to 600 (balanced across the
three classes) to keep training fast on a free Colab T4 GPU.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


LABEL_MAP = {0: "bearish", 1: "bullish", 2: "neutral"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/samples.jsonl")
    ap.add_argument("--n", type=int, default=600,
                    help="Total samples (balanced across 3 classes)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    per_class = args.n // 3

    print("Loading zeroshot/twitter-financial-news-sentiment from HuggingFace...")
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    # Combine train + validation, then re-split ourselves.
    all_rows = list(ds["train"]) + list(ds["validation"])
    print(f"  Loaded {len(all_rows)} total examples")

    # Bucket by class
    buckets = {0: [], 1: [], 2: []}
    for row in all_rows:
        buckets[row["label"]].append(row["text"])
    print(f"  Class counts in source: "
          f"bearish={len(buckets[0])}, "
          f"bullish={len(buckets[1])}, "
          f"neutral={len(buckets[2])}")

    # Balanced subsample
    samples = []
    for label_id, texts in buckets.items():
        random.shuffle(texts)
        for text in texts[:per_class]:
            # Light cleanup: strip URLs that bloat tokenization without
            # adding signal. Keep ticker symbols ($AAPL) — they're useful.
            text = " ".join(t for t in text.split() if not t.startswith("http"))
            text = text.strip()
            if 10 < len(text) < 500:  # skip empty / huge
                samples.append({"text": text, "label": LABEL_MAP[label_id]})

    random.shuffle(samples)
    n = len(samples)
    for i, s in enumerate(samples):
        if i < int(n * 0.70):
            s["split"] = "train"
        elif i < int(n * 0.85):
            s["split"] = "val"
        else:
            s["split"] = "test"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"\nWrote {n} samples to {args.output}")
    print("Label distribution:", dict(Counter(s["label"] for s in samples)))
    print("Split distribution:", dict(Counter(s["split"] for s in samples)))


if __name__ == "__main__":
    main()
