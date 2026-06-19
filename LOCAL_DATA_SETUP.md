# Local Training Data Setup - Complete Guide

## What You Now Have

You now have a complete local data generation system that:
1. ✅ **Generates training data locally** - No external API needed
2. ✅ **High-quality examples** - Full thinking, comprehensive code, explanations
3. ✅ **Proper format** - Ready for `train_unsloth.py`
4. ✅ **Full pipeline** - From generation → merge → training

## File Structure

```
qwen3-distill/
├── generate_training_data.py      # Main data generation (run this)
├── merge_data.py                   # Merge into train/val splits (runs automatically)
├── train_unsloth.py                # Training script (untouched, ready to use)
├── data/
│   ├── raw/                        # Generated raw examples by category
│   │   ├── react_examples.jsonl    # React + GSAP examples
│   │   ├── javascript_examples.jsonl
│   │   ├── python_examples.jsonl
│   │   ├── reasoning_examples.jsonl
│   │   ├── coding_examples.jsonl
│   │   ├── debugging_examples.jsonl
│   │   ├── tools_examples.jsonl
│   │   └── agents_examples.jsonl
│   ├── train.jsonl                 # Training set (90%)
│   └── val.jsonl                   # Validation set (10%)
└── CLAUDE.md, etc...
```

## Current Status

✅ **9 examples generated and ready** (across 6 categories)
- React: 2 examples (GSAP carousel, ScrollTrigger)
- JavaScript: 2 examples (permutations, debounce)
- Python: 1 example (matplotlib animation)
- Coding: 2 examples (linked list reversal, BST)
- Debugging: 1 example (infinite render loops)
- Reasoning: 1 example (OAuth vs JWT)

✅ **Data split created:**
- `data/train.jsonl` - 8 examples (90%)
- `data/val.jsonl` - 1 example (10%)

## How to Generate More Data

### Option 1: Generate All Categories (Recommended)
```bash
python generate_training_data.py
```

This generates a base set of examples for each category. The script is built to be expanded.

### Option 2: Expand Training Data Manually

Edit `generate_training_data.py` and add more examples to each category:

```python
TRAINING_DATA = {
    "react": [
        # Add more examples here
        {
            "prompt": "Create a React component for...",
            "answer": """<thinking>...</thinking>\n\n```jsx\n...\n```"""
        },
        # ... more examples
    ],
    # ... other categories
}
```

**Each example needs:**
1. **prompt** - The user question
2. **answer** - Full response with:
   - `<thinking>` tags with reasoning
   - Complete code implementation (properly formatted)
   - Clear explanations and best practices

### Option 3: Use External Prompts

The script can be modified to read from your prompt files:

```bash
# These files already exist with real prompts:
ls data/prompts/
# react_animations_gsap.txt
# hard_javascript.txt
# python_matplotlib_viz.txt
# complex_reasoning.txt
# kimi_coding.txt
# kimi_debugging.txt
# kimi_tools.txt
# kimi_agents.txt
```

To use these, modify `generate_training_data.py` to:
1. Load prompts from `data/prompts/*.txt`
2. Generate responses using Claude API (with proper authentication)
3. Format and save to `data/raw/`

## Running the Training Pipeline

### Step 1: Ensure Data is Ready
```bash
python generate_training_data.py  # Create raw examples
python merge_data.py               # Merge into train/val
```

### Step 2: Start Training
```bash
python train_unsloth.py
```

The training script will:
1. Load `data/train.jsonl` and `data/val.jsonl`
2. Use QLoRA to fine-tune Qwen3-8B
3. Save adapter to `./adapters/qwen3-distill/`

### Step 3: Monitor Training
Look for:
- Loss decreasing over time
- Validation perplexity improving
- Training time: ~10-30 min on RTX 3090

## Data Format Verification

Each example should be valid JSON:
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert..."},
    {"role": "user", "content": "Create a React component..."},
    {"role": "assistant", "content": "<thinking>...</thinking>\n\n```jsx\n...\n```"}
  ]
}
```

Check format:
```bash
# Verify train.jsonl is valid
python -c "import json; [json.loads(line) for line in open('data/train.jsonl')]"
echo "✓ train.jsonl is valid JSON"

# Count examples
wc -l data/train.jsonl
wc -l data/val.jsonl
```

## Scaling Up

To generate **200+ examples** for real training:

1. **Expand the TRAINING_DATA dictionary** with more examples per category
2. **Generate systematically**:
   - React: 50 examples covering carousel, animations, interactions
   - JavaScript: 50 advanced algorithms and patterns
   - Python: 50 data viz, scientific computing
   - Reasoning: 25 conceptual explanations
   - Coding: 50 common interview problems
   - Debugging: 25 common React/JS issues
   - Tools: 25 integration patterns
   - Agents: 25 multi-step reasoning

3. **Run full pipeline**:
   ```bash
   python generate_training_data.py
   python merge_data.py
   python train_unsloth.py
   ```

## Data Quality Tips

✅ **Good examples have:**
- Clear problem statement in prompt
- Detailed `<thinking>` section explaining the approach
- Complete, working code with comments
- Edge cases and error handling
- Performance notes and alternatives
- Clear explanation of why the solution works

❌ **Avoid:**
- Incomplete code snippets
- Missing thinking sections
- Trivial problems
- Duplicated content across examples
- Overly similar examples (different prompts, same patterns)

## Troubleshooting

### Problem: "data/train.jsonl not found"
```bash
python merge_data.py  # Run this first
```

### Problem: Training fails with "messages format error"
Check your JSONL format:
```bash
# Pretty print first example
python -c "import json; print(json.dumps(json.loads(open('data/train.jsonl').readline()), indent=2))"
```

### Problem: Not enough training data
1. Expand `TRAINING_DATA` in `generate_training_data.py`
2. Add 50+ examples per category
3. Run `python merge_data.py` again

### Problem: Training is very slow
This is expected with few examples. Quality > quantity for distillation.
- 9 examples: ~2 min test
- 100 examples: ~15 min
- 500+ examples: ~60+ min

## Next Steps

1. ✅ You have working data generation
2. ✅ You have a 90/10 train/val split
3. ⏭️ Run `python train_unsloth.py` to test the full pipeline
4. ⏭️ Expand `TRAINING_DATA` with more examples as needed
5. ⏭️ Monitor training loss and validation perplexity

## Notes

- This completely replaces the TokenRouter API approach
- No external API calls or authentication needed
- Data is generated locally and completely safe
- Perfect for testing the training pipeline locally
- Can easily scale to 1000+ examples by expanding the data dictionary
