#!/usr/bin/env python3
"""
Parallel data generation for Qwen3-8B distillation.
Run this in separate terminals with different categories.

Usage:
  Terminal 1: python generate_data_parallel.py react
  Terminal 2: python generate_data_parallel.py javascript
  Terminal 3: python generate_data_parallel.py python
  Terminal 4: python generate_data_parallel.py reasoning
  Terminal 5: python generate_data_parallel.py coding
  Terminal 6: python generate_data_parallel.py debugging
  Terminal 7: python generate_data_parallel.py tools
  Terminal 8: python generate_data_parallel.py agents

Then run: python merge_data.py
"""

import os, json, sys, time, random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

TOKENROUTER_API_KEY = os.environ.get("TOKENROUTER_API_KEY")
if not TOKENROUTER_API_KEY:
    raise ValueError("TOKENROUTER_API_KEY not set in .env file")

client = OpenAI(
    base_url="https://api.tokenrouter.com/v1",
    api_key=TOKENROUTER_API_KEY
)

SYSTEM_PROMPT = """You are an expert software engineer with deep knowledge of React, JavaScript, Python, animations, and data visualization.
Think through problems step by step inside <thinking>...</thinking> tags, explaining your reasoning and approach.
Then give your final answer with clean, well-structured code and detailed explanations.
Be thorough, correct, and explain complex concepts clearly."""

TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search for current information or documentation",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "run_bash",
        "description": "Execute a command",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}
        }, "required": ["command"]}
    }},
]

# Category definitions
CATEGORIES = {
    "react": ("React + GSAP Animations", "react_animations_gsap.txt", 250, False),
    "javascript": ("Hard JavaScript", "hard_javascript.txt", 250, True),
    "python": ("Python + Matplotlib", "python_matplotlib_viz.txt", 250, False),
    "reasoning": ("Complex Reasoning", "complex_reasoning.txt", 250, True),
    "coding": ("Coding (Original)", "kimi_coding.txt", 250, False),
    "debugging": ("Debugging", "kimi_debugging.txt", 250, False),
    "tools": ("Tool Use", "kimi_tools.txt", 250, True),
    "agents": ("Agent Tasks", "kimi_agents.txt", 250, True),
}

def load_prompts(filename):
    path = Path(f"data/prompts/{filename}")
    if not path.exists():
        print(f"⚠️  Warning: {path} not found")
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def generate_example(prompt, use_tools=False):
    # Available models via TokenRouter:
    # - "minimax/MiniMax-M3" (currently using)
    # - "moonshotai/kimi-k2.7-code" (Kimi K2)
    # - "x-ai/grok-build-0.1" (Grok from X)
    try:
        kwargs = {
            "model": "minimax/MiniMax-M3",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 1,
            "max_tokens": 3000,
        }
        if use_tools:
            kwargs["tools"] = TOOLS
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message
    except Exception as e:
        print(f"Error: {e}")
        return None

def to_jsonl_entry(user_prompt, assistant_msg):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
        tool_calls_list = [
            {"type": "function", "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }} for tc in assistant_msg.tool_calls
        ]
        messages.append({
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": tool_calls_list
        })
    else:
        messages.append({"role": "assistant", "content": assistant_msg.content})
    return {"messages": messages}

def generate_category(category_name):
    if category_name not in CATEGORIES:
        print(f"❌ Unknown category: {category_name}")
        print(f"Available: {', '.join(CATEGORIES.keys())}")
        sys.exit(1)

    cat_display, prompt_file, count, use_tools = CATEGORIES[category_name]

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    output_file = Path(f"data/raw/{category_name}_examples.jsonl")

    print(f"\n{'='*60}")
    print(f"Generating {cat_display}")
    print(f"{'='*60}")

    prompts = load_prompts(prompt_file)
    if not prompts:
        print(f"❌ No prompts found for {category_name}")
        return 0

    prompts = prompts[:count]
    print(f"Loaded {len(prompts)} prompts from {prompt_file}")

    generated = 0
    with open(output_file, "w") as f:
        for prompt in tqdm(prompts, desc=cat_display):
            msg = generate_example(prompt, use_tools=use_tools)
            if msg and msg.content:
                entry = to_jsonl_entry(prompt, msg)
                f.write(json.dumps(entry) + "\n")
                generated += 1
            time.sleep(0.5)  # Rate limit

    print(f"\n✅ Generated {generated} examples → {output_file}")
    return generated

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_data_parallel.py <category>")
        print(f"Categories: {', '.join(CATEGORIES.keys())}")
        sys.exit(1)

    category = sys.argv[1].lower()
    generate_category(category)
