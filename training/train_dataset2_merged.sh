#!/bin/bash
# Merge datasets 1 & 2, then train Qwen3-8B with combined data

set -e

echo "============================================================"
echo "Merging Datasets 1 & 2 for training"
echo "============================================================"

# Step 1: Merge both datasets
echo ""
echo "Step 1: Merging train.jsonl + train2.jsonl..."

python3 << 'MERGE_SCRIPT'
import json
from pathlib import Path

data_dir = Path("data")
mlx_dir = data_dir / "mlx_data"

all_train = []
all_val = []

# Load dataset 1
with open(data_dir / "train.jsonl") as f:
    for line in f:
        all_train.append(json.loads(line))
with open(data_dir / "val.jsonl") as f:
    for line in f:
        all_val.append(json.loads(line))

# Load dataset 2
with open(data_dir / "train2.jsonl") as f:
    for line in f:
        all_train.append(json.loads(line))
with open(data_dir / "val2.jsonl") as f:
    for line in f:
        all_val.append(json.loads(line))

print(f"✅ Merged: {len(all_train)} training + {len(all_val)} validation examples")

# Write to mlx_data
mlx_dir.mkdir(parents=True, exist_ok=True)

with open(mlx_dir / "train.jsonl", "w") as f:
    for ex in all_train:
        f.write(json.dumps(ex) + "\n")

with open(mlx_dir / "valid.jsonl", "w") as f:
    for ex in all_val:
        f.write(json.dumps(ex) + "\n")

print(f"✅ Written to data/mlx_data/")
MERGE_SCRIPT

# Step 2: Train with merged data
echo ""
echo "Step 2: Training LoRA adapter on merged datasets..."
echo "  - 245 training examples"
echo "  - 2800 iterations (~11 epochs)"
echo ""

python -m mlx_lm lora \
    --model Qwen/Qwen3-8B \
    --data data/mlx_data \
    --adapter-path ./adapters/qwen3-mlx-dataset2 \
    --iters 2800 \
    --batch-size 1 \
    --learning-rate 2e-5 \
    --train

echo "✅ LoRA training complete!"

# Step 3: Merge adapter with base model
echo ""
echo "Step 3: Merging adapter with base model..."
python -m mlx_lm fuse \
    --model Qwen/Qwen3-8B \
    --adapter-path ./adapters/qwen3-mlx-dataset2 \
    --save-path ./adapters/qwen3-merged-dataset2

echo "✅ Model merged!"

echo ""
echo "============================================================"
echo "✅ Training complete with merged datasets!"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  • LoRA adapter: ./adapters/qwen3-mlx-dataset2/"
echo "  • Merged model: ./adapters/qwen3-merged-dataset2/"
echo ""
