#!/bin/bash
# Merge all datasets and launch mlx-lm LoRA training
set -e

cd /Users/nr/Developer/labs/qwen3-distill

echo "============================================================"
echo "Qwen3-8B Training Pipeline — $(date)"
echo "============================================================"

# Step 1: Regenerate dataset2 JSONL (in case it wasn't saved)
echo ""
echo "Step 1: Regenerating dataset2 JSONL..."
python3 generate_dataset2.py
echo "✅ dataset2 generated"

# Step 2: Merge datasets into mlx_data format
echo ""
echo "Step 2: Merging all datasets..."
python3 - <<'PYEOF'
import json, os, random
from pathlib import Path

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

# Load all data
d1_train = load_jsonl('data/train.jsonl')
d1_val   = load_jsonl('data/val.jsonl')
d2_train = load_jsonl('data/train2.jsonl')
d2_val   = load_jsonl('data/val2.jsonl')

train_all = d1_train + d2_train
val_all   = d1_val   + d2_val

random.seed(42)
random.shuffle(train_all)

out = Path('data/mlx_data_combined')
out.mkdir(exist_ok=True)

def write_jsonl(path, data):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

write_jsonl(out / 'train.jsonl', train_all)
write_jsonl(out / 'valid.jsonl', val_all)

print(f"✅ Combined: {len(train_all)} train + {len(val_all)} val")
print(f"   Saved to data/mlx_data_combined/")
PYEOF

# Step 3: Train with mlx-lm LoRA
echo ""
echo "Step 3: Starting mlx-lm LoRA training..."
python3 -m mlx_lm lora \
    --model Qwen/Qwen3-8B \
    --data data/mlx_data_combined \
    --adapter-path ./adapters/qwen3-mlx-v2 \
    --iters 3000 \
    --batch-size 1 \
    --learning-rate 2e-5 \
    --train \
    --save-every 500

echo "✅ LoRA training complete!"

# Step 4: Fuse adapter
echo ""
echo "Step 4: Fusing adapter..."
python3 -m mlx_lm fuse \
    --model Qwen/Qwen3-8B \
    --adapter-path ./adapters/qwen3-mlx-v2 \
    --save-path ./adapters/qwen3-merged-v2

echo "✅ Model fused!"

echo ""
echo "============================================================"
echo "✅ Training complete! — $(date)"
echo "   Adapter: ./adapters/qwen3-mlx-v2/"
echo "   Merged:  ./adapters/qwen3-merged-v2/"
echo "============================================================"
