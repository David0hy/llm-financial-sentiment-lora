"""LoRA SFT for financial sentiment classification.

Fine-tunes Qwen2.5-1.5B-Instruct with LoRA adapters on the Twitter Financial
News Sentiment dataset. Demonstrates the standard SFT setup with PEFT.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)

SYSTEM_PROMPT = """You are a financial sentiment classifier. Given a finance-related tweet, \
classify the sentiment into exactly one of three categories: bearish, bullish, or neutral.
Respond with ONLY the category name in lowercase."""


def load_split(path, split):
    with open(path) as f:
        return [json.loads(line) for line in f if json.loads(line)["split"] == split]


def format_example(ex, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Classify the sentiment of this tweet:\n\n{ex['text']}\n\nSentiment:"},
        {"role": "assistant", "content": ex["label"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"formatted_text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/samples.jsonl")
    ap.add_argument("--out", default="checkpoints/lora_r16")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--quant", action="store_true", help="Use 4-bit quantization (QLoRA)")
    args = ap.parse_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model (optionally 4-bit quantized for QLoRA)
    if args.quant:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=bnb, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.float16, device_map="auto"
        )

    # LoRA config
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # Should print ~0.5-0.7% trainable

    # Data
    train = load_split(args.data, "train")
    val = load_split(args.data, "val")
    train_ds = Dataset.from_list(train).map(lambda x: format_example(x, tokenizer))
    val_ds = Dataset.from_list(val).map(lambda x: format_example(x, tokenizer))

    def tokenize(batch):
        return tokenizer(batch["formatted_text"], truncation=True, max_length=512, padding=False)

    train_ds = train_ds.map(tokenize, batched=True,
                            remove_columns=["formatted_text", "text", "label", "split"])
    val_ds = val_ds.map(tokenize, batched=True,
                        remove_columns=["formatted_text", "text", "label", "split"])

    # Training (fp16 for T4 compatibility — bf16 only works on A100/H100)
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
