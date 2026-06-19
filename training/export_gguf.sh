#!/bin/bash
# Export fused MLX model to GGUF for Ollama — run AFTER train_and_export.sh
set -e

MERGED="./adapters/qwen35-9b-v3-merged"
GGUF_OUT="./qwen35-9b-v3.gguf"
LLAMA_CPP="./llama.cpp"

echo "============================================================"
echo "Exporting to GGUF — $(date)"
echo "============================================================"

if [ ! -d "$LLAMA_CPP" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP"
    pip install -q -r "$LLAMA_CPP/requirements.txt"
fi

python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$MERGED" \
    --outtype q8_0 \
    --outfile "$GGUF_OUT"

echo "✅ GGUF → $GGUF_OUT"

# Register with Ollama
cat > Modelfile <<EOF
FROM $GGUF_OUT
SYSTEM """You are an expert software engineer with deep knowledge of Python, JavaScript, React, TypeScript, animations, AI/LLM systems, and distributed architectures. Think through problems step by step inside <thinking>...</thinking> tags."""
EOF

ollama create qwen3-distill -f Modelfile \
    && echo "✅ Run with: ollama run qwen3-distill" \
    || echo "⚠️  Ollama not installed — install from ollama.com then re-run"
