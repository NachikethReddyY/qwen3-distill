#!/bin/bash
# Resume Qwen3-8B LoRA training from iteration 1000 checkpoint

set -e

echo "============================================================"
echo "Resuming Qwen3-8B LoRA training from iteration 1000"
echo "============================================================"

# Resume LoRA training from checkpoint
echo ""
echo "Step 1: Resuming LoRA adapter training..."
python -m mlx_lm lora \
    --model Qwen/Qwen3-8B \
    --data data/mlx_data \
    --adapter-path ./adapters/qwen3-mlx \
    --resume-adapter-file ./adapters/qwen3-mlx/0001000_adapters.safetensors \
    --iters 1800 \
    --batch-size 1 \
    --learning-rate 2e-5 \
    --train

echo "✅ LoRA training complete!"

# Step 2: Merge adapter with base model
echo ""
echo "Step 2: Merging adapter with base model..."
python -m mlx_lm fuse \
    --model Qwen/Qwen3-8B \
    --adapter-path ./adapters/qwen3-mlx \
    --save-path ./adapters/qwen3-merged-mlx

echo "✅ Model merged!"

echo ""
echo "============================================================"
echo "✅ Training pipeline complete!"
echo "============================================================"
