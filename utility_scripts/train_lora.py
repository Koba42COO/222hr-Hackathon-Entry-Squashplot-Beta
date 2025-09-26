#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

"""
Minimal LoRA trainer for instruction-tuning on your dataset.
Pick a base model you have access to (examples below).
"""

EXAMPLES = {
  "mistral": "mistralai/Mistral-7B-v0.1",
  "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
  "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
}

def build_prompt(example):
    # ChatML-like: system + user + assistant
    msgs = example["conversations"]
    text=""
    for m in msgs:
        role = m["role"]
        if role=="system":
            text += f"<|system|>\n{m['content']}\n"
        elif role=="user":
            text += f"<|user|>\n{m['content']}\n"
        elif role in ("assistant","model"):
            text += f"<|assistant|>\n{m['content']}\n"
    text += "<|end|>\n"
    return {"text": text}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=EXAMPLES["mistral"], help="HF model id or local path")
    ap.add_argument("--data", default="data/aiva_convos.jsonl")
    ap.add_argument("--out",  default="outputs/aiva-lora")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fp8", action="store_true")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(build_prompt)

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    tok.pad_token = tok.eos_token

    def tok_fn(ex):
        return tok(ex["text"], truncation=True, max_length=2048)

    ds_tok = ds.map(tok_fn, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype="auto",
        device_map="auto"
    )

    # LoRA config (adjust for different model architectures)
    # Try different target modules based on model type
    if "gpt" in args.base.lower() or "dialo" in args.base.lower():
        # GPT-style models
        target_modules = ["c_attn", "c_proj"] if "c_attn" in [n for n, _ in model.named_modules()] else ["attn.c_attn", "attn.c_proj"]
    else:
        # Standard transformer modules
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

    # Filter to only modules that exist in the model
    available_modules = [n for n, _ in model.named_modules()]
    target_modules = [m for m in target_modules if any(m in mod for mod in available_modules)]

    if not target_modules:
        # Fallback to all linear layers
        target_modules = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and 'lm_head' not in n][:8]

    print(f"Using target modules: {target_modules}")

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args_tr = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds_tok,
        data_collator=collator
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"✅ LoRA adapter saved to: {args.out}")

if __name__=="__main__":
    main()