#!/usr/bin/env python3
"""
Generate training data for Qwen3-8B distillation using Kimi K2 API.
Includes React/GSAP, hard JavaScript, Python/matplotlib, and complex reasoning examples.
"""

import os, json, time, random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

KIMI_API_KEY = os.environ.get("KIMI_API_KEY")
if not KIMI_API_KEY:
    raise ValueError("KIMI_API_KEY not set in .env file")

client = OpenAI(
    base_url="https://api.moonshot.ai/v1",
    api_key=KIMI_API_KEY
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

def load_prompts(filename):
    path = Path(f"data/prompts/{filename}")
    if not path.exists():
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def generate_example(prompt, use_tools=False):
    try:
        kwargs = {
            "model": "kimi-k2.7-code",
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

def generate_dataset():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    output_file = Path("data/raw/kimi_examples.jsonl")

    print("Loading prompts...")
    datasets = [
        ("React + GSAP Animations", load_prompts("react_animations_gsap.txt")[:20], False),
        ("Hard JavaScript", load_prompts("hard_javascript.txt")[:25], True),
        ("Python + Matplotlib", load_prompts("python_matplotlib_viz.txt")[:20], False),
        ("Complex Reasoning", load_prompts("complex_reasoning.txt")[:20], True),
        ("Coding (Original)", load_prompts("kimi_coding.txt")[:20], False),
        ("Debugging", load_prompts("kimi_debugging.txt")[:15], False),
        ("Tool Use", load_prompts("kimi_tools.txt")[:15], True),
        ("Agent Tasks", load_prompts("kimi_agents.txt")[:15], True),
    ]

    total = sum(len(prompts) for _, prompts, _ in datasets)
    print(f"Generating {total} examples from Kimi K2 (your languages!)...")

    generated = 0
    with open(output_file, "w") as f:
        for category, prompts, use_tools in datasets:
            print(f"\n{category} ({len(prompts)} examples):")
            for prompt in tqdm(prompts, desc=category):
                msg = generate_example(prompt, use_tools=use_tools)
                if msg and msg.content:
                    entry = to_jsonl_entry(prompt, msg)
                    f.write(json.dumps(entry) + "\n")
                    generated += 1
                time.sleep(0.5)

    print(f"\n✅ Generated {generated} examples → {output_file}")
    return generated

def split_train_val():
    raw_file = Path("data/raw/kimi_examples.jsonl")
    train_file = Path("data/train.jsonl")
    val_file = Path("data/val.jsonl")

    examples = []
    with open(raw_file) as f:
        for line in f:
            examples.append(json.loads(line))

    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)

    with open(train_file, "w") as f:
        for ex in examples[:split_idx]:
            f.write(json.dumps(ex) + "\n")

    with open(val_file, "w") as f:
        for ex in examples[split_idx:]:
            f.write(json.dumps(ex) + "\n")

    print(f"\n✅ Split data:")
    print(f"   Train: {len(examples[:split_idx])} examples")
    print(f"   Val: {len(examples[split_idx:])} examples")

if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3-8B with YOUR Languages (React, JS, Python, GSAP)")
    print("=" * 60)
    count = generate_dataset()
    if count > 0:
        print("\nSplitting into train/val...")
        split_train_val()
        print("\n✅ Ready for Phase 3 training!")
    else:
        print("\n❌ No examples generated.")
