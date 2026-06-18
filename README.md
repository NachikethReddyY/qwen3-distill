# Qwen3-8B Dev Engine

Building a fast, coding-focused agentic model via knowledge distillation from Kimi K2.

## Overview

```
Kimi K2 (API, your $10 credits)
  ↓
Generate ~7,000 training examples
  ↓
Train Qwen3-8B (Nosana RTX 3090, ~$2.30)
  ↓
Convert to GGUF
  ↓
Upload to HuggingFace
  ↓
Run in LM Studio on your Mac (40-60 tok/s, 2-3x faster than 12B)
```

## Quick Start

### 1. Setup (30 min)
```bash
cd ~/Developer/labs/qwen3-distill
bash setup.sh
nano .env  # Add your Kimi API key and HF token
```

### 2. Generate Data (6-8 hrs, ~$8-10)
```bash
source venv/bin/activate
python generate_data_kimi.py
```

This calls Kimi K2 API ~7,000 times to generate:
- Coding problems + solutions
- Debugging + refactoring examples
- Tool calling scenarios (web search, terminal, etc.)
- Multi-step agent tasks

Data saved to:
- `data/train.jsonl` (6,300 examples)
- `data/val.jsonl` (700 examples)

### 3. (Optional) Add More Data from Gemma 4 (8-10 hrs, ~$2-3)
After Phase 2 completes, decide if you want to add hard algorithm examples from Gemma 4 26B on Nosana.

### 4. Train (12 hrs, ~$2.30)
Upload `data/train.jsonl` and `data/val.jsonl` to Nosana, run `train_unsloth.py` on RTX 3090.

### 5. Convert to GGUF (30 min)
Merge LoRA adapter + quantize to Q4_K_M and Q5_K_M.

### 6. Upload to HuggingFace (10 min)
Push GGUF files to your HuggingFace repo.

### 7. Test in LM Studio (1 hr)
Download from HuggingFace, load in LM Studio, benchmark vs Gemma 4 12B.

## Files

| File | Purpose |
|---|---|
| `PLAN.md` | Detailed implementation plan |
| `setup.sh` | Environment setup script |
| `generate_data_kimi.py` | Call Kimi K2 API, generate training data |
| `.env` | API keys (never commit) |
| `data/prompts/` | Source prompts for each category |
| `data/train.jsonl` | Generated training examples |
| `data/val.jsonl` | Generated validation examples |

## Expected Results

| Metric | Gemma 4 12B (now) | Qwen3-8B distilled (target) |
|---|---|---|
| Speed on 32GB Mac | ~15-20 tok/s | ~40-60 tok/s |
| Memory | ~8-10GB | ~5-6GB |
| Coding (HumanEval) | ~74% | ~82-86% |
| Tool use | Good | Excellent |

## Cost Breakdown

| Phase | Cost | How |
|---|---|---|
| Data from Kimi | ~$8-10 | Use your Kimi K2 credits |
| Training on GPU | ~$2.30 | Use your Nosana credits |
| **Total** | **~$10** | All from credits you have |

## Status

- [x] Setup
- [ ] Phase 1A: Generate data from Kimi K2
- [ ] Phase 1B: (Optional) Generate data from Gemma 4
- [ ] Phase 3: Train on Nosana GPU
- [ ] Phase 4: Convert to GGUF
- [ ] Phase 5: Upload to HuggingFace
- [ ] Phase 6: Test in LM Studio

## Troubleshooting

**"KIMI_API_KEY not set"**
→ Make sure `.env` is created and has your Kimi API key

**"ModuleNotFoundError: No module named 'openai'"**
→ Run `source venv/bin/activate` then `pip install openai`

**Rate limited by Kimi API**
→ The script includes `time.sleep(0.5)` between calls. If you hit limits, try increasing it.

**Out of memory during training**
→ Reduce batch size: `--per_device_train_batch_size 1` and add `--gradient_checkpointing`

## Next Steps

1. Edit `.env` with your Kimi API key
2. Run `bash setup.sh`
3. Run `python generate_data_kimi.py`
4. Check results: `head -5 data/train.jsonl`
5. Proceed to Phase 3 training

See `PLAN.md` for detailed instructions.
