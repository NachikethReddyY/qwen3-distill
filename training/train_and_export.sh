#!/bin/bash
# Full pipeline: merge data → LoRA train → fuse MLX → export GGUF
# Hardware: MacBook Air M4 32GB — using 8-bit model
set -e

# ── Config ────────────────────────────────────────────────────
# Verify this model ID exists on HuggingFace mlx-community before running
MODEL="mlx-community/Qwen3.5-9B-8bit"
ADAPTER="./adapters/qwen35-9b-v3"
MERGED="./adapters/qwen35-9b-v3-merged"
ITERS=1200
BATCH=1

echo "============================================================"
echo "Qwen3 Training Pipeline — $(date)"
echo "Model: $MODEL  |  Iters: $ITERS  |  Batch: $BATCH"
echo "============================================================"

# ── Step 1: Merge all datasets ────────────────────────────────
echo ""
echo "Step 1: Merging all datasets..."
python3 - <<'PY'
import json, random
from pathlib import Path

def load(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

train = (load('data/train.jsonl') + load('data/train2.jsonl') +
         load('data/train3.jsonl') + load('data/train_tools.jsonl') +
         load('data/train_agents.jsonl') + load('data/train_tooluse.jsonl'))
val   = (load('data/val.jsonl')   + load('data/val2.jsonl')   +
         load('data/val3.jsonl')   + load('data/val_tools.jsonl') +
         load('data/val_agents.jsonl') + load('data/val_tooluse.jsonl'))

random.seed(42)
random.shuffle(train)

out = Path('data/mlx_data_combined')
out.mkdir(exist_ok=True)
(out / 'train.jsonl').write_text('\n'.join(json.dumps(x) for x in train) + '\n')
(out / 'valid.jsonl').write_text('\n'.join(json.dumps(x) for x in val)   + '\n')

epochs = round(ITERS / len(train), 1) if (ITERS := 1200) else 0
print(f"✅ Merged: {len(train)} train + {len(val)} val (~{epochs} epochs at 1200 iters)")
PY

# ── Step 2: LoRA training ─────────────────────────────────────
echo ""
echo "Step 2: Training LoRA adapter..."
python3 -m mlx_lm lora \
    --model "$MODEL" \
    --data data/mlx_data_combined \
    --adapter-path "$ADAPTER" \
    --iters $ITERS \
    --batch-size $BATCH \
    --grad-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --steps-per-report 50 \
    --steps-per-eval 400 \
    --save-every 200 \
    --max-seq-length 1024 \
    --grad-checkpoint \
    --train

echo "✅ LoRA training complete → $ADAPTER"

# ── Step 3: Fuse adapter → MLX model ─────────────────────────
echo ""
echo "Step 3: Fusing adapter into MLX model..."
python3 -m mlx_lm fuse \
    --model "$MODEL" \
    --adapter-path "$ADAPTER" \
    --save-path "$MERGED"

echo "✅ MLX fused model → $MERGED"

echo ""
echo "============================================================"
echo "✅ Training + fuse done — $(date)"
echo "   MLX adapter : $ADAPTER"
echo "   MLX fused   : $MERGED"
echo ""
echo "Next: run export_gguf.sh to convert to GGUF for Ollama"
echo "============================================================"
