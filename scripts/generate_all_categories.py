#!/usr/bin/env python3
"""
Generate training data for all categories using a specified model via TokenRouter.
Usage: python generate_all_categories.py minimax
       python generate_all_categories.py kimi
       python generate_all_categories.py grok
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

# Model mapping (from TokenRouter)
MODELS = {
    "minimax": "MiniMax-M3",
    "kimi": "moonshotai/kimi-k2.7-code",
    "grok": "x-ai/grok-build-0.1",
}

# All 8 categories
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

def generate_example(prompt, model_id, use_tools=False):
    try:
        kwargs = {
            "model": model_id,
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

def generate_all_with_model(model_name):
    if model_name not in MODELS:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available: {', '.join(MODELS.keys())}")
        sys.exit(1)

    model_id = MODELS[model_name]
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    output_file = Path(f"data/raw/{model_name}_all_categories.jsonl")

    print(f"\n{'='*60}")
    print(f"Generating ALL CATEGORIES with {model_name.upper()}")
    print(f"Model: {model_id}")
    print(f"{'='*60}\n")

    total_generated = 0

    with open(output_file, "w") as f:
        for cat_key, (cat_display, prompt_file, count, use_tools) in CATEGORIES.items():
            prompts = load_prompts(prompt_file)
            if not prompts:
                print(f"⚠️  No prompts for {cat_key}, skipping...")
                continue

            prompts = prompts[:count]
            print(f"\n{cat_display} ({len(prompts)} examples):")

            for prompt in tqdm(prompts, desc=cat_display):
                msg = generate_example(prompt, model_id, use_tools=use_tools)
                if msg and msg.content:
                    entry = to_jsonl_entry(prompt, msg)
                    f.write(json.dumps(entry) + "\n")
                    total_generated += 1
                time.sleep(0.5)  # Rate limit

    print(f"\n✅ Generated {total_generated} examples with {model_name.upper()} → {output_file}")
    return total_generated

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_all_categories.py <model>")
        print(f"Models: {', '.join(MODELS.keys())}")
        sys.exit(1)

    model = sys.argv[1].lower()
    generate_all_with_model(model)
