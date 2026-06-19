# Data Generation Scripts

This directory contains all scripts for generating and managing training data.

## Quick Start

```bash
# Regenerate merged datasets
python scripts/merge_data.py

# This creates:
# - data/mlx_data_combined/train.jsonl (331 examples)
# - data/mlx_data_combined/valid.jsonl (33 examples)
```

---

## Script Reference

### Data Generators

| Script | Output | Examples | Purpose |
|--------|--------|----------|---------|
| `generate_training_data.py` | `data/train.jsonl` | 180 | React, JS, Python, animations |
| `generate_dataset2.py` | `data/train2.jsonl` | 65 | Advanced React, TypeScript, CSS |
| `generate_dataset3.py` | `data/train3.jsonl` | 11 | Memory management, async Python |
| `generate_tooluse_data.py` | `data/train_tooluse.jsonl` | 29 | Multi-turn tool-use traces |
| Predefined (not generated) | `data/train_tools.jsonl` | 23 | Docker, REST, GraphQL |
| Predefined (not generated) | `data/train_agents.jsonl` | 23 | System design, architecture |

### Utilities

| Script | Purpose |
|--------|---------|
| `merge_data.py` | Merge all train*.jsonl files into mlx_data_combined/ |
| `add_examples.py` | Programmatically add new examples to datasets |
| `coding_interview_examples.py` | Generate coding interview questions |
| `conceptual_examples.py` | Generate reasoning/explanation examples |
| `debugging_examples.py` | Generate debugging scenario examples |
| `javascript_examples.py` | Generate JavaScript code examples |
| `python_distill_examples.py` | Generate Python code examples |
| `react_training_examples.py` | Generate React examples |
| `tools_integration_examples.py` | Generate tool integration examples |

---

## Workflow: Adding New Data

### 1. Generate New Examples
```bash
# Edit one of the generators to add more examples
# For example, add more React patterns to:
vim scripts/generate_training_data.py

# Then regenerate:
python scripts/generate_training_data.py
# Output: data/train.jsonl (updated)
```

### 2. Merge All Datasets
```bash
python scripts/merge_data.py
# Output:
# - data/mlx_data_combined/train.jsonl (all examples)
# - data/mlx_data_combined/valid.jsonl (10% split)
```

### 3. Train
```bash
./training/train_mlx.sh  # or train_nosana.py
```

---

## Data Format

Each JSONL line is a training example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert React developer..."
    },
    {
      "role": "user",
      "content": "Create a React component that..."
    },
    {
      "role": "assistant",
      "content": "<thinking>\nStep 1: Understand requirements\nStep 2: Plan implementation\nStep 3: Write code\n</thinking>\n\n```jsx\nconst MyComponent = () => {\n  return <div>...</div>\n}\n```\n\n**Key points:**\n- Explanation of approach\n- Performance notes\n- When to use this pattern"
    }
  ]
}
```

**Requirements:**
- `messages` array with role/content pairs
- Roles: `system`, `user`, `assistant`
- Reasoning in `<thinking>` tags (not `<think>`)
- Code in markdown blocks with language (jsx, python, js, etc.)
- Clear explanations of solution

---

## Adding Your Own Examples

### Manual (Quick)
```bash
cat >> data/train.jsonl << 'EOF'
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "<thinking>...</thinking>\n..."}]}
EOF
```

### Programmatic (Recommended)
```python
import json
from pathlib import Path

new_examples = [
    {
        "messages": [
            {"role": "system", "content": "You are an expert..."},
            {"role": "user", "content": "Your prompt here"},
            {"role": "assistant", "content": "<thinking>\nYour reasoning\n</thinking>\n\n```language\nYour code\n```\n\nYour explanation"}
        ]
    }
]

with open("data/train.jsonl", "a") as f:
    for ex in new_examples:
        f.write(json.dumps(ex) + "\n")

print(f"Added {len(new_examples)} examples")
```

Then rebuild:
```bash
python scripts/merge_data.py
```

---

## Generator Details

### generate_training_data.py
- **Creates:** React, JS, Python, coding, reasoning examples
- **Hardcoded:** 180 examples total
- **Edit to add:** More examples in the TRAINING_DATA dict
- **Output:** data/train.jsonl

### generate_dataset2.py
- **Creates:** Advanced React, TypeScript, CSS, Web APIs
- **Hardcoded:** 65 examples
- **Edit to add:** More frontend patterns
- **Output:** data/train2.jsonl

### generate_dataset3.py
- **Creates:** Memory management, Python async, edge cases
- **Hardcoded:** 11 examples (batch 1 only)
- **TODO:** Batch 2 (JS animations), Batch 3 (Python advanced)
- **Output:** data/train3.jsonl

### generate_tooluse_data.py
- **Creates:** Multi-turn tool-use traces
- **Examples:** Tool calling, function composition, agent loops
- **Hardcoded:** 29 examples (3 batches done)
- **TODO:** Batch 4 (skill chaining, memory-augmented loops)
- **Output:** data/train_tooluse.jsonl

---

## Tips

- **Always validate JSON:** `python -c "import json; [json.loads(l) for l in open('data/train.jsonl')]"`
- **Check example count:** `wc -l data/train.jsonl`
- **Verify all files:** `for f in data/train*.jsonl; do echo "$(wc -l < $f) $(basename $f)"; done`
- **Keep examples diverse:** Mix difficulty levels, languages, domains
- **Quality > quantity:** 300 good examples > 1000 poor ones
- **Always include reasoning:** `<thinking>` tags help model learn to reason

---

## Next Steps

1. Run `python scripts/merge_data.py` to prepare data
2. Run training: `./training/train_mlx.sh` or `python training/train_nosana.py`
3. Test output quality
4. If not satisfied, add more examples and retrain

