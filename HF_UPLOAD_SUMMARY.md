# HuggingFace Upload Summary

**Date:** June 19, 2026  
**Repository:** https://huggingface.co/nachikethreddyy/qwen3.5-8b-distilled

## Repository Structure

Successfully organized and uploaded Qwen3.5-8B distilled model variants to HuggingFace with clean folder structure:

```
nachikethreddyy/qwen3.5-8b-distilled/
├── full_precision/           (16.39 GB - BF16 safetensors)
│   ├── model-00001-of-00004.safetensors
│   ├── model-00002-of-00004.safetensors
│   ├── model-00003-of-00004.safetensors
│   ├── model-00004-of-00004.safetensors
│   ├── model.safetensors.index.json
│   ├── config.json
│   ├── generation_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
│
├── mlx/4bit/                 (30.5 GB - MLX format, Apple Silicon optimized)
│   ├── model.safetensors
│   ├── config.json
│   ├── generation_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
│
├── variants/q8/              (8.8 GB - INT8 quantized)
│   ├── model.safetensors
│   ├── config.json
│   ├── generation_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
│
├── gguf/                     (15.3 GB - GGUF F16 format)
│   └── qwen3-distilled-f16.gguf
│
└── README.md                 (Comprehensive usage guide)
```

## Model Variants

| Variant | Location | Size | Format | Best For |
|---------|----------|------|--------|----------|
| Full Precision | `full_precision/` | 16.39 GB | Safetensors (BF16) | Maximum quality, research, fine-tuning |
| Q8 Quantized | `variants/q8/` | 8.8 GB | Safetensors (INT8) | Production inference, GPU memory constrained |
| MLX 4-bit | `mlx/4bit/` | 30.5 GB | MLX format | Apple Silicon (M1/M2/M3/M4) inference |
| GGUF F16 | `gguf/` | 15.3 GB | GGUF | Local inference (Ollama, llama.cpp, LM Studio) |

## Upload Method

Used `huggingface_hub.upload_folder()` for direct upload without git-lfs, which is the recommended approach for large model repositories.

## Quick Start Examples

### Transformers (Any OS)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "nachikethreddyy/qwen3.5-8b-distilled",
    device_map="auto"
)
```

### MLX (Mac)
```python
from mlx_lm import load, generate
model, tokenizer = load("nachikethreddyy/qwen3.5-8b-distilled")
```

### Ollama
```bash
ollama run nachikethreddyy/qwen3.5-8b-distilled:F16
```

### llama.cpp
```bash
llama-cli -hf nachikethreddyy/qwen3.5-8b-distilled:F16
```

## Upload Details

- **Upload Time:** ~10 minutes
- **Total Size:** ~55 GB across all variants
- **Status:** ✅ Complete and accessible
- **Repository Privacy:** Public
- **Git Method:** Direct upload via huggingface_hub (not git-based due to large file sizes)

## Local Reference

Local copy: `/Users/nr/Developer/labs/qwen3-distill/qwen3.5-8b-distilled-clean/`

This local folder mirrors the HuggingFace repository structure for reference and local testing.
