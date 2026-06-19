# 🚀 READY TO TRAIN

Your environment is fully set up. Here's what's configured:

## ✅ Setup Complete

- **API Key**: Nosana API key configured in `.env`
- **Virtual Env**: Python 3.14.6 with all dependencies installed
- **Training Data**: 302 examples (train) + 33 examples (validation)
- **Model**: Qwen3.5-9B (will auto-download from HuggingFace)
- **GPU**: NVIDIA RTX 5090 ($0.40/h)

## 📊 Training Configuration

- **Model**: Qwen/Qwen3.5-9B
- **Training samples**: 302
- **Validation samples**: 33
- **Batch size**: 4
- **Gradient accumulation**: 2
- **Max seq length**: 1024
- **Iterations**: ~1,200 (≈3-4 epochs)
- **LoRA config**: r=16, alpha=32

## 🎯 When Ready, Run This

### Option 1: Submit to Nosana (GPU, ~30-45 min)
```bash
cd /Users/nr/Developer/labs/qwen3-distill
source .venv/bin/activate
python training/train_nosana.py
```

### Option 2: Local Training (M4 Mac, slower)
```bash
cd /Users/nr/Developer/labs/qwen3-distill
cd training
bash merge_and_train.sh
```

## 📝 What Happens

1. Qwen3.5-9B model downloads from HuggingFace (~6GB)
2. 4-bit quantization applied
3. LoRA adapters trained (saves to `./adapters/qwen35-9b-v3-gpu/`)
4. Checkpoints saved every 200 steps
5. Training logs printed to console

## 📤 After Training: Push Results

```bash
cd /Users/nr/Developer/labs/qwen3-distill
git add adapters/ data/
git commit -m "Train Qwen3 v2: 302 examples, RTX 5090"
git push origin main
```

## ⏱️ Expected Time

- **Nosana RTX 5090**: 30-45 minutes
- **Local M4 Mac**: 2-3 hours

---

**Just let me know when you want to start! 🎯**
