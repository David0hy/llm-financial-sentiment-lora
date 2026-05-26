"""Run inference with a trained LoRA adapter on the financial sentiment task."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

SYSTEM_PROMPT = """You are a financial sentiment classifier. Given a finance-related tweet, \
classify the sentiment into exactly one of three categories: bearish, bullish, or neutral.
Respond with ONLY the category name in lowercase."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/lora_r16.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    with open(args.data) as f:
        samples = [json.loads(line) for line in f if json.loads(line)["split"] == "test"]

    preds = []
    for s in tqdm(samples, desc="lora inference"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify the sentiment of this tweet:\n\n{s['text']}\n\nSentiment:"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        preds.append({"text": s["text"], "label": s["label"], "pred": gen.strip().lower()})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
