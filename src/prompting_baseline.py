"""Zero-shot prompting baseline for financial sentiment classification.

Supports two backends:
  - 'qwen': local Qwen2.5-1.5B-Instruct (apples-to-apples with the LoRA model)
  - 'gpt-4o-mini': OpenAI API (ceiling check — a much larger model)

Outputs predictions to JSON for unified evaluation by eval.py.
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

SYSTEM_PROMPT = """You are a financial sentiment classifier. Given a finance-related tweet, \
classify the sentiment expressed in it into exactly one of three categories:

- bearish: the tweet expresses a negative outlook on a stock, market, or economic indicator
- bullish: the tweet expresses a positive outlook
- neutral: the tweet is factual, informational, or non-directional

Respond with ONLY the category name in lowercase: bearish, bullish, or neutral."""

USER_TEMPLATE = "Classify the sentiment of this tweet:\n\n{text}\n\nSentiment:"


def load_test_samples(path):
    with open(path) as f:
        return [json.loads(line) for line in f if json.loads(line)["split"] == "test"]


def predict_qwen(samples, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    preds = []
    for s in tqdm(samples, desc="qwen zero-shot"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=s["text"])},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        preds.append({"text": s["text"], "label": s["label"], "pred": gen.strip().lower()})
    return preds


def predict_openai(samples, model="gpt-4o-mini"):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    preds = []
    for s in tqdm(samples, desc=f"{model} zero-shot"):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(text=s["text"])},
            ],
            temperature=0,
            max_tokens=8,
        )
        pred = resp.choices[0].message.content.strip().lower()
        preds.append({"text": s["text"], "label": s["label"], "pred": pred})
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "gpt-4o-mini"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or f"results/{args.model}_zero_shot.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    samples = load_test_samples(args.data)
    print(f"Predicting on {len(samples)} test samples...")

    if args.model == "qwen":
        preds = predict_qwen(samples)
    else:
        preds = predict_openai(samples, model=args.model)

    with open(out_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
