# Qwen3.5-9B Fine-Tuning for Claude Code

High-quality fine-tuned model for reasoning, code generation, and tool use. Trained on **331 curated examples** across React, Python, JavaScript, system design, and tool integration.

**Goal:** A small, fast model that writes production code and reasons like Claude while being runnable locally.

---

## Quick Start

### 1. Training on Local Mac (M3/M4)
```bash
cd /Users/nr/Developer/labs/qwen3-distill
./training/train_mlx.sh
```
**Time:** 1-2 hours | **Memory:** 32GB | **Output:** MLX adapter

### 2. Training on GPU (Nosana/CUDA)
```bash
pip install -r training/requirements-gpu.txt
python training/train_nosana.py
```
**Time:** 30 min - 2 hrs (varies by GPU) | **Cost:** $0.25-1.00/hr | **Output:** PyTorch adapter

### 3. Test the Trained Model
```bash
# After either training finishes:
python -m mlx_lm generate \
  --model ./adapters/qwen35-9b-v3-merged \
  --prompt "Write a React component that..." \
  --max-tokens 500
```

---

## Project Structure

```
qwen3-distill/
├── README.md                          # This file
├── data/
│   ├── train.jsonl                    # 180 examples (React, JS, Python, animations)
│   ├── train2.jsonl                   # 65 examples (Advanced React, TypeScript)
│   ├── train3.jsonl                   # 11 examples (Memory, async, edge cases)
│   ├── train_tools.jsonl              # 23 examples (Docker, REST, GraphQL)
│   ├── train_agents.jsonl             # 23 examples (System design, architecture)
│   ├── train_tooluse.jsonl            # 29 examples (Multi-turn harness tool-use)
│   ├── mlx_data_combined/             # Merged train+val datasets (ready for training)
│   └── prompts/                       # Raw prompts by category
├── scripts/
│   ├── generate_*.py                  # Data generators for each category
│   ├── merge_data.py                  # Merge datasets into training sets
│   └── [utility scripts]
├── training/
│   ├── train_mlx.sh                   # Local M4 MacBook training
│   ├── train_nosana.py                # GPU/Nosana training (PyTorch)
│   ├── export_gguf.sh                 # Convert to GGUF for Ollama
│   ├── requirements-gpu.txt           # GPU dependencies
│   ├── README.md                      # Detailed training guide
│   └── [other training scripts]
├── adapters/
│   ├── qwen35-9b-v3/                  # LoRA adapter (after training)
│   ├── qwen35-9b-v3-merged/           # Fused model (MLX)
│   └── qwen35-9b-v3-gpu/              # Adapter from GPU training
└── models/                            # GGUF exports (after conversion)
```

---

## Training Workflow: MLX vs GPU

| Feature | MLX (Local Mac) | GPU (Nosana) |
|---------|-----------------|--------------|
| **Speed** | Baseline | 2-6x faster |
| **Cost** | Free | $0.25-1.00/hr |
| **Memory required** | 32GB | 8GB+ VRAM |
| **OOM risk** | Higher | Lower |
| **Setup** | Ready to go | `pip install` + API key |
| **Output** | MLX model | PyTorch adapter |

Choose **MLX** for testing on your Mac. Choose **GPU** if you hit OOM or need production speed.

---

## Training Commands

### Local Mac (MLX)
```bash
./training/train_mlx.sh
```

### GPU / Nosana
```bash
pip install -r training/requirements-gpu.txt
python training/train_nosana.py
```

### Export to GGUF
```bash
./training/export_gguf.sh
```

See `training/README.md` for detailed setup and troubleshooting.

---

## Dataset (331 examples)

- **React/JavaScript** (122): Hooks, animations, performance, TypeScript
- **Python** (45): Async, data viz, decorators, fast APIs
- **System Design** (23): Microservices, databases, scalability
- **Tool Integration** (52): Multi-turn tool-use, Docker, REST, GraphQL
- **Coding** (15): Algorithms, interview problems
- **Reasoning** (24): Conceptual explanations, debugging, edge cases

All examples include reasoning traces (`<thinking>` tags) and production patterns.

---

## Testing the Model

### MLX
```bash
python -m mlx_lm generate \
  --model ./adapters/qwen35-9b-v3-merged \
  --prompt "Write a React hook that..." \
  --max-tokens 500
```

### GGUF/Ollama
```bash
ollama run qwen3-distill "Explain OAuth vs JWT"
```

### PyTorch
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B")
model = PeftModel.from_pretrained(model, "./adapters/qwen35-9b-v3-gpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")

inputs = tokenizer("Write a React hook", return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
print(tokenizer.decode(outputs[0]))
```

---

## Troubleshooting

### Training OOM
- **Mac:** Close other apps, reduce `--max-seq-length` to 512, try GPU
- **GPU:** Reduce batch size to 1, reduce seq-length to 512, use cheaper GPU

### Data not found
```bash
cd /Users/nr/Developer/labs/qwen3-distill
python scripts/merge_data.py  # Rebuild merged datasets
```

See `training/README.md` for more troubleshooting.

---

## Next Steps

1. Choose training method (MLX or GPU)
2. Run training
3. Test generated output
4. Export to GGUF if needed
5. Deploy or iterate

See detailed guide in `training/README.md`.

