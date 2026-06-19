# Qwen3.5-9B Fine-Tuning

Two training approaches for the same dataset:

## 1. Local MLX Training (M4/M3 Mac)

**Best for:** Quick iteration, testing, 32GB+ RAM

```bash
cd /Users/nr/Developer/labs/qwen3-distill
./training/train_mlx.sh
```

**What it does:**
- Merges all datasets (train.jsonl, train2.jsonl, train3.jsonl, train_tools.jsonl, train_agents.jsonl, train_tooluse.jsonl)
- Trains LoRA adapter on MLX (Apple Silicon optimized)
- Fuses into MLX model for inference
- Output: `./adapters/qwen35-9b-v3-merged/`

**Hardware requirements:**
- M3/M4/M5 MacBook
- 32GB unified memory minimum
- Uses: 8-bit quantization, batch-size 1, gradient checkpointing

**If OOM:**
- Reduce `--max-seq-length` to 512
- Use `--num-layers 8` to reduce LoRA scope
- Check Activity Monitor for other processes

---

## 2. GPU Training (CUDA / Nosana)

**Best for:** Production training, cost-effective scaling, avoiding OOM

```bash
# Local CUDA GPU (RTX 3090, A100, etc.)
cd /Users/nr/Developer/labs/qwen3-distill
pip install -r training/requirements-gpu.txt
python training/train_nosana.py

# Or on Nosana cloud GPU:
nosana jobs submit \
  --image nosana/pytorch:cuda-12.1 \
  --gpu nvidia-a100 \
  --command "pip install -r training/requirements-gpu.txt && python training/train_nosana.py"
```

**What it does:**
- Merges same datasets (uses `data/mlx_data_combined/`)
- Trains LoRA adapter using transformers + bitsandbytes
- 4-bit quantization for GPU memory efficiency
- Output: `./adapters/qwen35-9b-v3-gpu/`

**Hardware requirements:**
- Any CUDA GPU (8GB+ VRAM)
- RTX 3090, A100, H100 recommended
- Or: Nosana cloud rental

**Nosana Setup:**
1. Create Nosana account at https://app.nosana.io/
2. Set up API key: `export NOSANA_API_KEY=your_key`
3. Choose GPU tier (RTX 3090, A100, H100)
4. Submit job (see command above)
5. Monitor: `nosana jobs list` and `nosana jobs logs <job-id>`

**Cost comparison:**
- Nosana RTX 3090: ~$0.25/hr
- Nosana A100: ~$0.50/hr
- Nosana H100: ~$1.00/hr

**Training time:**
- RTX 3090: ~1-2 hours (331 examples, 1200 iters)
- A100: ~30-45 minutes
- H100: ~20-30 minutes

---

## Dataset

Both scripts use the same merged dataset:

```
data/mlx_data_combined/
├── train.jsonl    # 280+ training examples
└── valid.jsonl    # 30+ validation examples
```

**Sources:**
- `train.jsonl` (180) - React, JS, Python, animations
- `train2.jsonl` (65) - Advanced React, TypeScript
- `train3.jsonl` (11) - Memory management, Python async
- `train_tools.jsonl` (23) - Docker, REST, GraphQL
- `train_agents.jsonl` (23) - System design, architecture
- `train_tooluse.jsonl` (29) - Multi-turn tool-use traces

---

## Adapter Merging & Export

After either training completes:

```bash
# Merge adapter into base model (for MLX)
python -m mlx_lm fuse \
  --model mlx-community/Qwen3.5-9B-8bit \
  --adapter-path ./adapters/qwen35-9b-v3 \
  --save-path ./adapters/qwen35-9b-v3-merged

# Export to GGUF (for Ollama)
python -m mlx_lm convert \
  --model ./adapters/qwen35-9b-v3-merged \
  --quantize 4 \
  --output-file ./models/qwen3-distill.gguf
```

---

## Testing the Fine-Tuned Model

```bash
# MLX inference
python -m mlx_lm generate \
  --model ./adapters/qwen35-9b-v3-merged \
  --prompt "Write a React hook for..." \
  --max-tokens 500

# Ollama inference (after GGUF export)
ollama run qwen3-distill "Write a React hook for..."
```

---

## When to Use Which?

| Scenario | Method |
|----------|--------|
| Testing on your Mac | MLX local |
| Quick iteration (< 2 hrs) | MLX local |
| Production training | GPU (Nosana or local) |
| Large dataset (1000+ ex) | GPU required |
| Memory errors | Switch to GPU |
| Cost-sensitive | Nosana RTX 3090 |
| Speed-critical | Nosana A100/H100 |

---

## Troubleshooting

### MLX OOM on Mac
- Reduce batch size to 1 (already done)
- Reduce max-seq-length to 512
- Close other apps, restart terminal
- Check `Activity Monitor` → Memory tab

### GPU OOM on Nosana
- Reduce batch size (4 → 2 → 1)
- Reduce max-seq-length to 512
- Enable `packing=True` in train_nosana.py
- Try smaller GPU first (RTX 3090 vs A100)

### Data not found
- Ensure you're in project root: `cd /Users/nr/Developer/labs/qwen3-distill`
- Run data merge first: see `scripts/` directory

---

## Next Steps

1. **Test locally**: `./training/train_mlx.sh` (or Python GPU locally)
2. **If satisfied**: Deploy to Nosana for production training
3. **After training**: Export to GGUF, test inference
4. **Integration**: Use fine-tuned model in your app

