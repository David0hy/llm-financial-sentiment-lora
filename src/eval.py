"""Unified evaluation across all approaches.

Reads prediction JSONs from results/, computes accuracy + per-class F1,
prints a comparison table.
"""

import argparse
import glob
import json

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


LABELS = ["bearish", "bullish", "neutral"]


def normalize(pred):
    pred = pred.strip().lower()
    for lab in LABELS:
        if lab in pred:
            return lab
    return "unknown"


def evaluate(pred_path):
    with open(pred_path) as f:
        preds = json.load(f)
    y_true = [p["label"] for p in preds]
    y_pred = [normalize(p["pred"]) for p in preds]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    return {"path": pred_path, "n": len(preds), "accuracy": acc,
            "macro_f1": report["macro avg"]["f1-score"],
            "per_class_f1": {lab: report[lab]["f1-score"] for lab in LABELS},
            "confusion_matrix": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pattern", default="results/*.json")
    args = ap.parse_args()

    results = []
    for p in sorted(glob.glob(args.pattern)):
        results.append(evaluate(p))

    print(f"\n{'Approach':<40} {'N':>5} {'Acc':>7} {'Macro F1':>10}")
    print("-" * 70)
    for r in results:
        name = r["path"].replace("results/", "").replace(".json", "")
        print(f"{name:<40} {r['n']:>5} {r['accuracy']:>7.3f} {r['macro_f1']:>10.3f}")

    print("\nPer-class F1:")
    for r in results:
        name = r["path"].replace("results/", "").replace(".json", "")
        f1s = "  ".join(f"{lab}={r['per_class_f1'][lab]:.2f}" for lab in LABELS)
        print(f"  {name}: {f1s}")


if __name__ == "__main__":
    main()
