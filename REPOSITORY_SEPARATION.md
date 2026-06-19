# Repository Separation Summary

**Date:** June 19, 2026

Successfully separated the Qwen3.5-8B distilled model into two focused HuggingFace repositories.

## 📦 Repository Structure

### 1. GGUF Repository
**URL:** https://huggingface.co/nachikethreddyy/qwen3.5-8b-distilled-GGUF

Best for: **Local inference** (Ollama, llama.cpp, LM Studio)

**Contents:**
- `gguf/` - GGUF F16 format (15.3 GB)
- `full_precision/` - Full precision Safetensors (16.39 GB)
- `variants/q8/` - Q8 quantized Safetensors (8.8 GB)
- `README.md` - GGUF-focused documentation

**Use Cases:**
```bash
# Ollama
ollama run nachikethreddyy/qwen3.5-8b-distilled-GGUF:F16

# llama.cpp
llama-cli -hf nachikethreddyy/qwen3.5-8b-distilled-GGUF:F16

# LM Studio (GUI)
Search and download in LM Studio interface
```

### 2. MLX Repository
**URL:** https://huggingface.co/nachikethreddyy/qwen3.5-8b-distilled-MLX

Best for: **Apple Silicon** (M1/M2/M3/M4)

**Contents:**
- `4bit/` - MLX optimized format (~31 GB effective)
- `README.md` - MLX-focused documentation

**Use Cases:**
```python
from mlx_lm import load, generate

model, tokenizer = load("nachikethreddyy/qwen3.5-8b-distilled-MLX")
response = generate(model, tokenizer, prompt="...", max_tokens=200)
```

## 🎯 Why Separate?

| Aspect | GGUF Repo | MLX Repo |
|--------|-----------|----------|
| **Target Users** | Cross-platform (any OS) | Apple Silicon only |
| **Primary Tools** | Ollama, llama.cpp, LM Studio, Transformers | MLX framework |
| **Installation** | Minimal (pre-built binaries) | Requires MLX library |
| **Performance** | Good across platforms | Excellent on Mac |
| **File Size** | More organized, multiple formats | Compact MLX format |

## 📋 File Organization

### GGUF Repo Structure
```
nachikethreddyy/qwen3.5-8b-distilled-GGUF/
├── gguf/                    ← Primary format
│   └── qwen3-distilled-f16.gguf (15.3 GB)
├── full_precision/          ← Alternative
│   ├── model-00001-of-00004.safetensors
│   ├── model-00002-of-00004.safetensors
│   ├── model-00003-of-00004.safetensors
│   ├── model-00004-of-00004.safetensors
│   └── ...
├── variants/q8/             ← Alternative
│   └── model.safetensors (8.8 GB)
└── README.md
```

### MLX Repo Structure
```
nachikethreddyy/qwen3.5-8b-distilled-MLX/
├── 4bit/                    ← Primary MLX format
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── ...
└── README.md
```

## ✅ What Was Done

1. ✅ Renamed original repo to `qwen3.5-8b-distilled-GGUF`
2. ✅ Created new `qwen3.5-8b-distilled-MLX` repository
3. ✅ Uploaded MLX files to MLX repo
4. ✅ Removed MLX folder from GGUF repo (eliminated duplicates)
5. ✅ Updated READMEs for each repo with format-specific guidance
6. ✅ Cross-linked repositories in documentation

## 🔗 Quick Links

- **GGUF:** https://huggingface.co/nachikethreddyy/qwen3.5-8b-distilled-GGUF
- **MLX:** https://huggingface.co/nachikethreddyy/qwen3.5-8b-distilled-MLX

## 💾 Local Mirror

Local copy still at: `/Users/nr/Developer/labs/qwen3-distill/qwen3.5-8b-distilled-clean/`

Can be deleted to save ~70GB disk space (all data backed up on HF).
