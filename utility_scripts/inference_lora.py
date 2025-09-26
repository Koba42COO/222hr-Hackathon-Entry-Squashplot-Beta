#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model id")
    ap.add_argument("--adapter", required=True, help="Path to LoRA output dir")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")

    print("AiVA (fine-tuned) ready. Type your message:")
    while True:
        try:
            user = input("> ").strip()
            if not user: continue
            prompt = f"<|system|>\nYou are AiVA.\n<|user|>\n{user}\n<|assistant|>\n"
            out = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"]
            # print only assistant tail
            print(out.split("<|assistant|>")[-1].split("<|end|>")[0].strip())
        except KeyboardInterrupt:
            break

if __name__=="__main__":
    main()
