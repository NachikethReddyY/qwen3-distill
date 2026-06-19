#!/bin/bash
# Train Qwen3-8B with mlx-lm LoRA, merge, and convert to GGUF

set -e

echo "============================================================"
echo "Training Qwen3-8B with mlx-lm LoRA on Apple Silicon"
echo "============================================================"

# Step 1: Train LoRA adapter with mlx-lm
echo ""
echo "Step 1: Training LoRA adapter..."
python -m mlx_lm lora \
    --model Qwen/Qwen3-8B \
    --data data/mlx_data \
    --adapter-path ./adapters/qwen3-mlx \
    --iters 2000 \
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

# Step 3: Convert to GGUF (optional - requires llama.cpp)
echo ""
echo "Step 3: Converting to GGUF format..."
echo "Note: This requires llama.cpp tools"

if command -v python &> /dev/null; then
    # Try to convert if llama.cpp convert script is available
    if [ -f "llama.cpp/convert_hf_to_gguf.py" ]; then
        python llama.cpp/convert_hf_to_gguf.py \
            ./adapters/qwen3-merged-mlx \
            --outtype q4_0 \
            -o qwen3-distill.gguf
        echo "✅ GGUF conversion complete! Output: qwen3-distill.gguf"
    else
        echo "⚠️  llama.cpp not found. To convert to GGUF manually:"
        echo "   1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
        echo "   2. Run: python llama.cpp/convert_hf_to_gguf.py ./adapters/qwen3-merged-mlx/ --outtype q4_0"
    fi
fi

echo ""
echo "============================================================"
echo "✅ Training pipeline complete!"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  • LoRA adapter: ./adapters/qwen3-mlx/"
echo "  • Merged model (HF format): ./adapters/qwen3-merged-mlx/"
echo "  • GGUF model: qwen3-distill.gguf (after conversion)"
echo ""
