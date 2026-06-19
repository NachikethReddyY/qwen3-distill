# Local API Data Generation

This replaces the external TokenRouter API with local Claude generation. Perfect for testing your training pipeline without external API dependencies.

## Quick Start

### 1. Generate a Single Category (Production-Grade Quality)
```bash
# Generate 250 React examples (high quality, with full thinking)
python generate_local_api_data.py react 250

# Or any category
python generate_local_api_data.py javascript 250
python generate_local_api_data.py python 250
python generate_local_api_data.py reasoning 250
```

### 2. Generate All Categories at Once
```bash
python generate_local_api_data.py all 250
```

This will generate 250 examples for each of these categories:
- **react** - React + GSAP Animations
- **javascript** - Hard JavaScript Problems
- **python** - Python + Matplotlib Visualizations
- **reasoning** - Complex Reasoning Tasks
- **coding** - General Coding Problems
- **debugging** - Debugging Challenges
- **tools** - Tool Use and Integration
- **agents** - Agent Task Planning

### 3. Merge Data into Train/Val Splits
```bash
python merge_data.py
```

This creates:
- `data/train.jsonl` (90% of examples)
- `data/val.jsonl` (10% of examples)

### 4. Train Your Model
```bash
python train_unsloth.py
```

## Data Quality

Each generated example includes:
- **Detailed thinking** - Step-by-step reasoning in `<thinking>` tags
- **Complete code** - Full, working implementations
- **Best practices** - Error handling, edge cases, performance notes
- **Clear explanations** - Thorough descriptions of concepts

## Output Format

Each example is JSONL with this structure:
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert..."},
    {"role": "user", "content": "Create a React component with..."},
    {"role": "assistant", "content": "<thinking>...</thinking>\n\n```jsx\n...\n```"}
  ]
}
```

This matches the training format expected by `train_unsloth.py`.

## How Many Examples Do You Need?

- **Quick test**: 10-20 per category (~100 total) - validates pipeline
- **Reasonable training**: 100-250 per category (~1000 total) - good model
- **Full training**: 500+ per category - professional results

## Files Generated

- `data/raw/{category}_examples.jsonl` - Raw examples per category
- `data/train.jsonl` - Training split (after merging)
- `data/val.jsonl` - Validation split (after merging)

## Troubleshooting

### If generation is slow:
- This is expected - Claude is generating comprehensive, high-quality examples
- Each example takes 2-5 seconds plus API latency
- For 250 examples: ~10-20 minutes

### If you get API errors:
- Ensure your `ANTHROPIC_API_KEY` is set
- Check you have Claude API access
- See `https://console.anthropic.com/`

### To use a faster model (lower quality):
Edit the script and change:
```python
model="claude-opus-4-8"  # Current: best quality
model="claude-sonnet-4-6"  # Faster, still good quality
```

## Development Workflow

```bash
# 1. Quick validation (2 examples per category)
python generate_local_api_data.py react 5
python merge_data.py
python train_unsloth.py  # Should work if format is correct

# 2. Actual training (full dataset)
python generate_local_api_data.py all 250
python merge_data.py
python train_unsloth.py
```

## Notes

- This script replaces `generate_data_parallel.py` and the TokenRouter API
- Uses Claude Opus 4.8 for production-grade training data
- Generates real, full-thinking examples (not test stubs)
- Safe for educational testing of the entire training pipeline
