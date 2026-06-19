#!/usr/bin/env python3
"""
Local API data generation for Qwen3-8B distillation.
Replaces external API calls with Claude as the generation source.
Generates training data directly without TokenRouter dependency.

Usage:
  python generate_local_api_data.py react 10          # 10 examples for React
  python generate_local_api_data.py javascript 5      # 5 examples for JS
  python generate_local_api_data.py all               # All categories (30 total)
  python generate_local_api_data.py test              # Quick test (2 per category)
"""

import os
import json
import sys
import time
from pathlib import Path
from anthropic import Anthropic

SYSTEM_PROMPT = """You are an expert software engineer with deep knowledge of React, JavaScript, Python, animations, and data visualization.
Think through problems step by step inside <thinking>...</thinking> tags, explaining your reasoning and approach.
Then give your final answer with clean, well-structured code and detailed explanations.
Be thorough, correct, and explain complex concepts clearly."""

# Category definitions matching generate_data_parallel.py
CATEGORIES = {
    "react": "react_animations_gsap.txt",
    "javascript": "hard_javascript.txt",
    "python": "python_matplotlib_viz.txt",
    "reasoning": "complex_reasoning.txt",
    "coding": "kimi_coding.txt",
    "debugging": "kimi_debugging.txt",
    "tools": "kimi_tools.txt",
    "agents": "kimi_agents.txt",
}

def load_prompts(filename, limit=None):
    """Load prompts from file."""
    path = Path(f"data/prompts/{filename}")
    if not path.exists():
        print(f"⚠️  Warning: {path} not found")
        return []
    with open(path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    if limit:
        prompts = prompts[:limit]
    return prompts

def generate_example(client, prompt):
    """Generate a comprehensive response using Claude API.

    Produces production-grade examples with:
    - Detailed thinking and reasoning
    - Complete code implementations
    - Thorough explanations
    - Best practices and edge cases
    """
    try:
        message = client.messages.create(
            model="claude-opus-4-8",  # Strongest model for best training data quality
            max_tokens=4000,
            temperature=0.7,  # Balanced creativity and coherence
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"""{prompt}

Please provide a comprehensive, production-grade response with:
1. Detailed thinking and reasoning inside <thinking></thinking> tags
2. Complete, working code with proper error handling
3. Clear explanations of key concepts
4. Best practices and edge cases
5. Performance considerations where relevant"""}
            ]
        )
        return message.content[0].text
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return None

def to_jsonl_entry(user_prompt, assistant_content):
    """Convert prompt and response to JSONL format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def generate_category(client, category_name, num_examples=None):
    """Generate examples for a specific category."""
    if category_name not in CATEGORIES:
        print(f"❌ Unknown category: {category_name}")
        print(f"Available: {', '.join(CATEGORIES.keys())}")
        return 0

    prompt_file = CATEGORIES[category_name]

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    output_file = Path(f"data/raw/{category_name}_examples.jsonl")

    print(f"\n{'='*60}")
    print(f"Generating {category_name.upper()}")
    print(f"{'='*60}")

    prompts = load_prompts(prompt_file, limit=num_examples)
    if not prompts:
        print(f"❌ No prompts found for {category_name}")
        return 0

    print(f"Loaded {len(prompts)} prompts from {prompt_file}")

    generated = 0
    with open(output_file, "w") as f:
        for i, prompt in enumerate(prompts, 1):
            print(f"  [{i}/{len(prompts)}] Generating...", end=" ", flush=True)
            response = generate_example(client, prompt)

            if response:
                entry = to_jsonl_entry(prompt, response)
                f.write(json.dumps(entry) + "\n")
                generated += 1
                print("✓")
            else:
                print("✗")

            # Small delay to avoid rate limiting
            time.sleep(0.5)

    print(f"✅ Generated {generated} examples → {output_file}")
    return generated

def main():
    if len(sys.argv) < 2:
        print("="*70)
        print("LOCAL API DATA GENERATION FOR QWEN3-8B DISTILLATION")
        print("="*70)
        print("\nUsage: python generate_local_api_data.py <category|all> [num_examples]")
        print(f"\nCategories: {', '.join(CATEGORIES.keys())}")
        print("\nExamples:")
        print("  python generate_local_api_data.py react 30       # 30 React examples")
        print("  python generate_local_api_data.py all            # All categories (default ~250 each)")
        print("  python generate_local_api_data.py javascript 50  # 50 JavaScript examples")
        print("\nThis script replaces the TokenRouter API with local Claude generation.")
        print("Generates high-quality, production-grade training data.")
        print("="*70)
        sys.exit(1)

    client = Anthropic()
    command = sys.argv[1].lower()
    num_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 250

    total_generated = 0

    if command == "all":
        print(f"\n📊 Generating all categories with {num_examples} examples each...")
        for category in CATEGORIES.keys():
            generated = generate_category(client, category, num_examples)
            total_generated += generated
    else:
        total_generated = generate_category(client, command, num_examples)

    print(f"\n{'='*60}")
    print(f"✅ TOTAL GENERATED: {total_generated} examples")
    print(f"{'='*60}")
    print("\n🔄 Next steps:")
    print("   1. python merge_data.py         # Merge data into train/val splits")
    print("   2. python train_unsloth.py      # Train the model")

if __name__ == "__main__":
    main()
