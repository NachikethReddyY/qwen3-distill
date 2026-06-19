# 🎓 Local Training Data Generation

**You now have a complete, self-contained data generation system for training Qwen3 locally.**

## What This Solves

✅ **No API dependency** - Generate data locally without TokenRouter or external services  
✅ **No authentication issues** - No keys, no config, just works  
✅ **For education/testing** - Perfect for testing your entire training pipeline  
✅ **Full control** - You control all data generation and quality  

## What You Have

### Files Created

```
generate_training_data.py    Main data generation script
add_examples.py              Helper to add/manage examples
merge_data.py                (existing) Merge into train/val splits
train_unsloth.py             (existing) Training script
STATUS.md                    Current status and quick reference
LOCAL_DATA_SETUP.md          Detailed setup guide
```

### Generated Data

```
data/
├── raw/                     Raw examples by category
│   ├── react_examples.jsonl (2 examples)
│   ├── javascript_examples.jsonl (2)
│   ├── python_examples.jsonl (1)
│   ├── coding_examples.jsonl (2)
│   ├── debugging_examples.jsonl (1)
│   └── reasoning_examples.jsonl (1)
├── train.jsonl              Training set (8 examples, 90%)
└── val.jsonl                Validation set (1 example, 10%)
```

## Quick Start

### 1️⃣ Test the Complete Pipeline
```bash
python train_unsloth.py
```

This will train on your 8 examples. Should complete in ~5-10 minutes.

### 2️⃣ Add More Examples (Optional)
```bash
python add_examples.py
# Interactive menu to add examples
```

Or edit `generate_training_data.py` directly to add more examples to `TRAINING_DATA` dictionary.

### 3️⃣ Regenerate Data
```bash
python generate_training_data.py
python merge_data.py
python train_unsloth.py
```

## Understanding the Data

Each training example has this structure:

```python
{
    "messages": [
        {"role": "system", "content": "You are an expert software engineer..."},
        {"role": "user", "content": "Create a React component that..."},
        {"role": "assistant", "content": "<thinking>...</thinking>\n\n```jsx\n...\n```\n\n**Explanation**..."}
    ]
}
```

**Key parts:**
1. **System message** - Defines the assistant's role and style
2. **User prompt** - The question/task
3. **Assistant response** - Should include:
   - `<thinking>` tags with step-by-step reasoning
   - Code block with complete implementation
   - Clear explanation and best practices

## Example Categories

### React (2 examples)
- GSAP carousel with timeline animations and stagger effects
- ScrollTrigger viewport animations

### JavaScript (2 examples)
- Finding all permutations of a string
- Debounce function with context preservation

### Python (1 example)
- Matplotlib animated visualization with multiple data series

### Coding (2 examples)
- Reverse linked list recursively
- Binary search tree with insert/search/delete

### Debugging (1 example)
- Fixing infinite render loops in React

### Reasoning (1 example)
- OAuth 2.0 vs JWT comparison and when to use each

## Adding More Examples

### Method 1: Interactive Mode
```bash
python add_examples.py
# Follow prompts to add new example
# Choose category, enter prompt, enter answer
```

### Method 2: Edit Python File
Edit `generate_training_data.py`:

```python
TRAINING_DATA = {
    "react": [
        # ... existing examples ...
        {
            "prompt": "Create a React component that animates...",
            "answer": """<thinking>
            I need to:
            1. Use useState for state management
            2. Use useEffect for animation setup
            3. Use GSAP for smooth animations
            </thinking>

            ```jsx
            import React, { useEffect, useState } from 'react';
            import gsap from 'gsap';

            export function MyComponent() {
              // Implementation here
            }
            ```

            **Key points:**
            - This uses GSAP for smooth animations
            - Memory-safe cleanup in useEffect
            - Responsive design"""
        }
    ]
}
```

Then regenerate:
```bash
python generate_training_data.py
python merge_data.py
```

## Scaling to Real Training

For professional training, you want 200-500+ examples.

### Recommended Distribution
- **React** (50 examples) - Various animation patterns
- **JavaScript** (50 examples) - Algorithm problems
- **Python** (50 examples) - Data science patterns  
- **Coding** (100 examples) - Interview-style problems
- **Reasoning** (50 examples) - Conceptual explanations
- **Debugging** (50 examples) - Common fixes
- **Tools** (50 examples) - Integration patterns
- **Agents** (50 examples) - Multi-step reasoning

### How to Add 200+ Examples
1. Create a script that reads from `data/prompts/*.txt`
2. For each prompt, generate a response using Claude/your LLM
3. Format and save to `data/raw/`
4. Run `merge_data.py` and train

Or manually add examples using `add_examples.py` repeatedly.

## Verification Checklist

✅ Data is valid JSON:
```bash
python -c "import json; [json.loads(l) for l in open('data/train.jsonl')]; print('✓')"
```

✅ Correct number of examples:
```bash
wc -l data/train.jsonl data/val.jsonl
```

✅ Messages have correct format:
```bash
python -c "import json; ex = json.loads(open('data/train.jsonl').readline()); print([m['role'] for m in ex['messages']])"
# Output should be: ['system', 'user', 'assistant']
```

✅ Training script can load data:
```bash
python -c "from datasets import load_dataset; load_dataset('json', data_files={'train': 'data/train.jsonl', 'validation': 'data/val.jsonl'}); print('✓')"
```

## How Data is Used in Training

1. **train_unsloth.py** loads `data/train.jsonl` and `data/val.jsonl`
2. Uses the "messages" field for SFT (Supervised Fine-Tuning)
3. The model learns to mimic the assistant responses
4. Each example teaches:
   - How to structure thinking clearly
   - How to write good code
   - How to explain concepts well
   - Best practices and common patterns

## Tips for High-Quality Data

✅ **DO:**
- Include detailed thinking sections
- Write complete, working code
- Explain why the solution works
- Include performance considerations
- Cover edge cases
- Show alternatives when relevant

❌ **DON'T:**
- Incomplete or pseudo-code
- Shallow explanations
- Trivial problems
- Duplicate content
- Poor formatting

## Troubleshooting

**Q: "ModuleNotFoundError: No module named 'datasets'"**
```bash
pip install datasets
```

**Q: "OSError: data/train.jsonl not found"**
```bash
python merge_data.py
```

**Q: "Invalid JSON in JSONL file"**
```bash
# Check format:
python -c "import json; json.loads(open('data/train.jsonl').readline())"
```

**Q: Training is very slow**
- This is expected! Quality training takes time
- Start with 9 examples to test
- Scale up as needed

**Q: How do I use this with a real API later?**
- Keep this system for testing locally
- For production, replace generation with API calls
- Same merge/train pipeline works either way

## Next Steps

1. ✅ Review STATUS.md for current state
2. ✅ Run `python train_unsloth.py` to test
3. ⏭️ Add more examples with `add_examples.py`
4. ⏭️ Regenerate and retrain
5. ⏭️ Scale to 200+ examples for real training

## Key Files

| File | Purpose |
|------|---------|
| `generate_training_data.py` | Generate examples from TRAINING_DATA dict |
| `add_examples.py` | Interactive menu to add/manage examples |
| `merge_data.py` | Merge raw examples into train/val split |
| `train_unsloth.py` | Train the model (unchanged) |
| `STATUS.md` | Quick reference and status |
| `LOCAL_DATA_SETUP.md` | Detailed setup guide |

## Summary

- ✅ Complete local data generation system
- ✅ 9 high-quality examples ready to train
- ✅ No external API or authentication needed
- ✅ Ready for end-to-end testing
- ✅ Easy to expand with more examples
- ✅ Perfect for educational purposes

**You can now train your model end-to-end locally!** 🚀
