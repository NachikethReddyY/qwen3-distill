#!/bin/bash
# Setup script for Qwen3-8B distillation project

set -e

echo "=========================================="
echo "Qwen3-8B Distillation - Setup"
echo "=========================================="

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install openai datasets huggingface_hub transformers tqdm python-dotenv

# Install llama.cpp for GGUF conversion
echo "Installing llama.cpp..."
if ! command -v llama-quantize &> /dev/null; then
    echo "Installing via Homebrew..."
    brew install llama.cpp
fi

# Create directories
echo "Creating project directories..."
mkdir -p data/prompts
mkdir -p data/raw
mkdir -p models
mkdir -p adapters
mkdir -p gguf
mkdir -p checkpoints

# Create .env template
if [ ! -f .env ]; then
    echo "Creating .env template..."
    cat > .env << 'EOF'
# Kimi K2 API Key from platform.moonshot.ai
KIMI_API_KEY=your_kimi_key_here

# HuggingFace token from huggingface.co/settings/tokens
HF_TOKEN=your_huggingface_token_here
EOF
    echo "⚠️  Created .env file. Please edit it with your API keys:"
    echo "   - KIMI_API_KEY: Get from platform.moonshot.ai"
    echo "   - HF_TOKEN: Get from huggingface.co/settings/tokens"
fi

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Make sure Kimi K2 API key has credits"
echo "3. Run: python generate_data_kimi.py"
echo ""
