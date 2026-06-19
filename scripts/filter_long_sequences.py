#!/usr/bin/env python3
"""Filter out sequences longer than max_tokens to prevent truncation."""

import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"
MAX_TOKENS = 1536
DATA_DIR = Path("data/mlx_data_combined")
OUTPUT_DIR = Path("data/mlx_data_filtered")

def count_tokens(text, tokenizer):
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=True))

def format_chat(messages):
    """Format messages as training text."""
    text_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            text_parts.append(f"[SYSTEM]\n{content}\n[/SYSTEM]\n")
        elif role == "user":
            text_parts.append(f"[USER]\n{content}\n[/USER]\n")
        elif role == "assistant":
            text_parts.append(f"[ASSISTANT]\n{content}\n[/ASSISTANT]\n")
    return "".join(text_parts)

def filter_data(input_file, output_file, tokenizer):
    """Filter sequences and save."""
    kept = 0
    removed = 0

    with open(input_file) as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            example = json.loads(line)
            messages = example.get("messages", [])
            text = format_chat(messages)
            tokens = count_tokens(text, tokenizer)

            if tokens <= MAX_TOKENS:
                f_out.write(json.dumps(example) + '\n')
                kept += 1
            else:
                removed += 1
                print(f"  Removed: {tokens} tokens (>{MAX_TOKENS})")

    return kept, removed

def main():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    OUTPUT_DIR.mkdir(exist_ok=True)

    for split in ['train', 'valid']:
        input_file = DATA_DIR / f"{split}.jsonl"
        output_file = OUTPUT_DIR / f"{split}.jsonl"

        print(f"\nFiltering {split}.jsonl (max {MAX_TOKENS} tokens)...")
        kept, removed = filter_data(input_file, output_file, tokenizer)

        print(f"  ✅ Kept: {kept}")
        print(f"  ❌ Removed: {removed}")
        print(f"  📁 Saved to: {output_file}")

if __name__ == "__main__":
    main()
    print(f"\n✅ Filtering complete! Use data/mlx_data_filtered/ for training.")
