# Qwen3-8B Dev Engine — Build Plan

## What We're Building
A fast, coding-focused agentic model that:
- Runs in LM Studio on your 32GB Mac at ~40-60 tok/s (vs ~15-20 tok/s for the 12B you have now)
- Works in any harness: Claude Code, Continue, Copilot, Ollama, etc.
- Native tool use / MCP / web search / terminal
- Upload to HuggingFace so anyone can download it

**Student model**: `Qwen/Qwen3-8B` (Apache 2.0, native tool calling, dual-mode thinking, best 8B GGUF support)
**Teachers**: Gemma 4 26B (local, free) + Kimi K2 (your $10 API credits)
**Training GPU**: Nosana RTX 3090 (your $10 Nosana credits, ~$2.30 per training run)

---

## Budget Summary

| What | Cost | How |
|---|---|---|
| Coding + reasoning training data | $0 | Gemma 4 26B in LM Studio (local) |
| Tool use + agent training data | ~$8-10 | Kimi K2 API (your existing credits) |
| GPU training (RTX 3090, ~12 hrs) | ~$2.30 | Nosana (your existing credits) |
| **Total out of pocket** | **~$0** | All covered by credits you have |

> **ChatGPT warning**: OpenAI's ToS Section 3 explicitly prohibits using API outputs
> to train models that compete with OpenAI. Do NOT use your $50 GPT credits for
> training data — it's a ToS violation.

> **Kimi K2 ToS**: Moonshot AI is generally permissive for research/personal model
> training. Verify at platform.moonshot.ai/terms before generating data.

---

## Architecture

```
┌────────────────────────────────────────┐
│           2 TEACHERS                   │
│                                        │
│  Gemma 4 26B MoE        Kimi K2        │
│  (LM Studio, local)     ($10 credits)  │
│  coding + reasoning     tool use +     │
│  ~5,000 examples        agent tasks    │
│                         ~5,000 examples│
└──────────┬──────────────────┬──────────┘
           │                  │
           └────────┬─────────┘
                    ▼
        Training Dataset (~10K JSONL)
        ├── coding problems + solutions
        ├── debugging + refactoring
        ├── tool calling (web, terminal, MCP)
        ├── agent task completions
        └── reasoning chains (<thinking> tags)
                    │
                    ▼
            Qwen3-8B (student base)
                    │
          QLoRA fine-tuning on Nosana
          RTX 3090, 24GB VRAM, Unsloth
                    │
                    ▼
        LoRA adapter (.safetensors)
                    │
              merge + quantize
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Q4_K_M.gguf            Q5_K_M.gguf
  → LM Studio             → HuggingFace
```

---

## Phase 0: Setup on Mac (~30 min)

**Working directory**: `~/Developer/labs/qwen3-distill/` (already created)

```bash
cd ~/Developer/labs/qwen3-distill

# Run setup script (installs venv, dependencies, creates directories)
bash setup.sh

# Edit .env with your API keys
nano .env   # (or edit in your editor)
```

**.env file needs:**
```
KIMI_API_KEY=your_kimi_key_here
HF_TOKEN=your_huggingface_token_here
```

**Get your API keys:**
- **Kimi K2**: Go to https://platform.moonshot.ai → API Keys → Create new key (copy it)
- **HuggingFace**: Go to https://huggingface.co/settings/tokens → New token → Copy it

**Verify setup:**
```bash
source venv/bin/activate
python -c "from openai import OpenAI; print('✅ OpenAI client imported')"
huggingface-cli login   # Enter your HF token when prompted
```

---

## Phase 1A: Kimi K2 API Data Generation (~$8-10 of your credits, ~6-8 hrs)

**Dataset from Kimi (~7,000-8,000 examples):**

| Category | Count | Why Kimi |
|---|---|---|
| Coding problems + solutions | 2,000 | Kimi strong at coding |
| Code debugging + refactoring | 1,500 | Debug reasoning excellent |
| Tool calling / function calls | 2,000 | Kimi's specialty: tool use |
| Multi-step agent tasks | 1,500 | SWE-bench quality agentic tasks |

**Run the data generation script:**

```bash
cd ~/Developer/labs/qwen3-distill
source venv/bin/activate

# This will call Kimi K2 API ~7,000 times and generate training data
# It takes ~6-8 hours (rate limited to avoid API throttling)
# Costs ~$8-10 of your Kimi K2 credits
python generate_data_kimi.py
```

**What it generates:**
- `data/raw/kimi_examples.jsonl` — All examples from Kimi
- `data/train.jsonl` — 6,300 training examples (90%)
- `data/val.jsonl` — 700 validation examples (10%)

**Monitoring:**
- Watch the progress bar — it should show ~7,000 examples
- Each category takes ~1-2 hours
- Press Ctrl+C to stop (you can resume later with partial data)

---

## Phase 1B: Gemma 4 26B on Nosana (Optional, ~$2-3 more, 8-10 hrs)

**Only run this if you want to add coding diversity after Phase 1A.**

If Kimi data quality looks good, you can skip this. But Gemma adds:
- Hard algorithms / competitive coding (2,000 examples)
- Complex reasoning chains (1,000 examples)

**Decision point after Phase 1A:**
1. Check the generated data: `head -20 data/train.jsonl`
2. If reasoning and tool use quality is high → skip Gemma, save credits
3. If you want more coding focus → proceed to Phase 1B on Nosana

We can set up Phase 1B after Phase 1A completes.

**`generate_data.py`** — calls both teachers:

```python
import os, json, time, random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Teacher 1: Gemma 4 26B via LM Studio
gemma = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Teacher 2: Kimi K2 via Moonshot API
kimi = OpenAI(
    base_url="https://api.moonshot.ai/v1",
    api_key=os.environ["KIMI_API_KEY"]
)

SYSTEM_PROMPT = """You are an expert software engineer and AI assistant.
Think through problems step by step inside <thinking>...</thinking> tags,
then give your final answer. For tool calls, output valid JSON in the correct format.
Be thorough, correct, and explain your reasoning clearly."""

def generate(client, model_name, user_prompt, tools=None):
    kwargs = dict(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    if tools:
        kwargs["tools"] = tools
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message

# Tool definitions (Qwen3 native format)
TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query"}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "run_bash",
        "description": "Execute a bash command in the terminal",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Bash command to run"}
        }, "required": ["command"]}
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the filesystem",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}
        }, "required": ["path"]}
    }},
]

def to_jsonl_entry(user_prompt, assistant_msg):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
        messages.append({
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": [
                {"type": "function", "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }} for tc in assistant_msg.tool_calls
            ]
        })
    else:
        messages.append({"role": "assistant", "content": assistant_msg.content})
    return {"messages": messages}

# Load source prompts (see data/prompts/ folder)
# Run in batches, save to data/raw/*.jsonl
# Then merge, filter, shuffle, split 90/10 into train.jsonl + val.jsonl
```

Source prompt files to create in `data/prompts/`:
- `coding_prompts.txt` — 200 coding problem prompts (from CodeAlpaca-20k)
- `debug_prompts.txt` — 150 broken code snippets to fix
- `tool_prompts.txt` — 250 tasks requiring web search / terminal / file access
- `agent_prompts.txt` — 250 multi-step dev workflow tasks

---

## Phase 2: Download Student Model (~20 min, Mac)

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/qwen3-8b
```

~16GB download. Keep this on Mac — you'll upload it to Nosana later.

---

## Phase 3: Train on Nosana GPU (~12 hrs, ~$2.30)

**GPU to select**: RTX 3090 ($0.192/hr, 24GB VRAM, 28 available)
- $10 credits ÷ $0.192 = ~52 hours available → ~4 full training runs if needed

**Nosana steps:**
1. Go to Nosana dashboard → New Job → select RTX 3090
2. Container: `docker.io/nosana/pytorch-jupyter:2.0.0`
3. Get JupyterLab URL from deployment
4. Upload via JupyterLab file browser:
   - `data/train.jsonl`
   - `data/val.jsonl`
   - `train_unsloth.py`
5. Open terminal in JupyterLab:

```bash
pip install unsloth
huggingface-cli login
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/qwen3-8b
python train_unsloth.py
```

**`train_unsloth.py`:**

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-8B",
    max_seq_length=8192,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)

dataset = load_dataset("json", data_files={
    "train": "data/train.jsonl",
    "validation": "data/val.jsonl"
})

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="messages",
    args=TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        fp16=True,
    ),
)

trainer.train()
model.save_pretrained("./adapters/qwen3-distill")
tokenizer.save_pretrained("./adapters/qwen3-distill")
print("Done. Download ./adapters/qwen3-distill/ via JupyterLab.")
```

6. Download `./adapters/qwen3-distill/` back to Mac via JupyterLab

**Mac fallback** (if Nosana has issues, free but slower ~16 hrs):
```bash
pip install mlx-lm
mlx_lm.lora --model ./models/qwen3-8b --train --data ./data \
  --iters 2000 --batch-size 2 --lora-layers 16 \
  --learning-rate 2e-5 --adapter-path ./adapters/qwen3-distill
```

---

## Phase 4: Merge + Convert to GGUF (~30 min, Mac)

```bash
# Merge LoRA adapter back into model weights
mlx_lm.fuse \
  --model ./models/qwen3-8b \
  --adapter-path ./adapters/qwen3-distill \
  --save-path ./models/qwen3-distill-merged

# Convert merged model to F16 GGUF
python $(brew --prefix llama.cpp)/convert_hf_to_gguf.py \
  ./models/qwen3-distill-merged \
  --outfile ./gguf/qwen3-distill-f16.gguf \
  --outtype f16

# Quantize: Q4_K_M = fastest in LM Studio
llama-quantize ./gguf/qwen3-distill-f16.gguf \
  ./gguf/qwen3-distill-Q4_K_M.gguf Q4_K_M

# Q5_K_M = better quality (for HuggingFace)
llama-quantize ./gguf/qwen3-distill-f16.gguf \
  ./gguf/qwen3-distill-Q5_K_M.gguf Q5_K_M
```

---

## Phase 5: Upload to HuggingFace (~10 min)

```bash
huggingface-cli repo create YOUR_USERNAME/qwen3-8b-devengine --type model

huggingface-cli upload YOUR_USERNAME/qwen3-8b-devengine \
  ./gguf/qwen3-distill-Q4_K_M.gguf \
  ./gguf/qwen3-distill-Q5_K_M.gguf \
  --repo-type model
```

---

## Phase 6: Test (~1 hr)

In LM Studio: search your HuggingFace repo name → download Q4_K_M → load → test.

**Benchmark checklist:**
- [ ] Write a quicksort in Python + explain time complexity
- [ ] Debug a broken async/await function
- [ ] Use web_search tool to find a library, then write code using it
- [ ] Multi-step terminal task: list files, read one, summarize content
- [ ] Compare tok/sec vs Gemma 4 12B — should be 2-3x faster

**Harness setup:**
- **Continue**: `~/.continue/config.json` → add as Ollama model (`ollama run YOUR_MODEL`)
- **Copilot**: Load via Ollama, set as completion provider
- **Claude Code**: Set as MCP tool model in settings

---

## Directory Structure

```
~/Developer/labs/qwen3-distill/
├── .env                    # API keys (never commit)
├── PLAN.md                 # This file
├── generate_data.py        # Data generation script
├── train_unsloth.py        # Training script (runs on Nosana)
├── convert.sh              # GGUF conversion script
├── data/
│   ├── prompts/            # Source prompt lists
│   │   ├── coding_prompts.txt
│   │   ├── debug_prompts.txt
│   │   ├── tool_prompts.txt
│   │   └── agent_prompts.txt
│   ├── raw/                # Raw teacher outputs
│   ├── train.jsonl         # 9,000 training examples
│   └── val.jsonl           # 1,000 validation examples
├── models/
│   ├── qwen3-8b/           # Downloaded base model
│   └── qwen3-distill-merged/  # After LoRA merge
├── adapters/
│   └── qwen3-distill/      # LoRA adapter from training
└── gguf/
    ├── qwen3-distill-Q4_K_M.gguf
    └── qwen3-distill-Q5_K_M.gguf
```

---

## Expected Results

| Metric | Gemma 4 12B (now) | Qwen3-8B distilled (target) |
|---|---|---|
| Speed on 32GB Mac | ~15-20 tok/s | ~40-60 tok/s |
| RAM usage | ~8-10GB | ~5-6GB |
| Coding (HumanEval) | ~74% | ~82-86% |
| Tool use quality | Good | Excellent (native Qwen3) |
| Context window | 256K | 128K |

---

## Risks

| Risk | Fix |
|---|---|
| Out of memory on Nosana RTX 3090 | Reduce batch size to 1, add `--gradient_checkpointing True` |
| Kimi ToS blocks training use | Switch to Devstral Small 2 (free Mistral API, Apache 2.0, 68% SWE-bench) |
| GGUF breaks tool calling format | Test with `llama-cli` before uploading to HuggingFace |
| Gemma 4 26B too slow in LM Studio | Reduce to Gemma 4 12B QAT for coding data (you already have it) |
| LM Studio doesn't recognize chat template | Add `tokenizer_config.json` with Qwen3 chat template to GGUF repo |
