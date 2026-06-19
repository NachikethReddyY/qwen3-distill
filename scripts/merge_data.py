#!/usr/bin/env python3
"""
Merge all parallel-generated data into train/val splits.
Run this after all parallel generation is complete.
"""

import json, random
from pathlib import Path

def merge_all_data():
    raw_dir = Path("data/raw")
    all_examples = []

    print("Merging all generated data...")

    # Find all *_examples.jsonl files
    jsonl_files = list(raw_dir.glob("*_examples.jsonl"))

    if not jsonl_files:
        print("❌ No generated data files found in data/raw/")
        return

    print(f"Found {len(jsonl_files)} category files:")
    for f in jsonl_files:
        print(f"  - {f.name}")

    # Load all examples
    total_loaded = 0
    for jsonl_file in jsonl_files:
        with open(jsonl_file) as f:
            count = 0
            for line in f:
                try:
                    example = json.loads(line)
                    all_examples.append(example)
                    count += 1
                except json.JSONDecodeError:
                    pass
        print(f"  Loaded {count} examples from {jsonl_file.name}")
        total_loaded += count

    print(f"\n✅ Total examples loaded: {total_loaded}")

    if total_loaded == 0:
        print("❌ No examples to merge!")
        return

    # Shuffle
    random.shuffle(all_examples)

    # Split 90/10
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    # Write train
    train_file = Path("data/train.jsonl")
    with open(train_file, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    # Write val
    val_file = Path("data/val.jsonl")
    with open(val_file, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n✅ Merged data:")
    print(f"   Train: {len(train_examples)} examples → {train_file}")
    print(f"   Val: {len(val_examples)} examples → {val_file}")
    print(f"\n🚀 Ready to train!")

if __name__ == "__main__":
    merge_all_data()
