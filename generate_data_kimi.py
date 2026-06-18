#!/usr/bin/env python3
"""
Generate training data for Qwen3-8B distillation using Kimi K2 API.
Generates ~7,000-8,000 examples of coding, debugging, tool use, and agent tasks.
"""

import os
import json
import time
import random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Initialize Kimi K2 client
KIMI_API_KEY = os.environ.get("KIMI_API_KEY")
if not KIMI_API_KEY:
    raise ValueError("KIMI_API_KEY not set in .env file")

client = OpenAI(
    base_url="https://api.moonshot.ai/v1",
    api_key=KIMI_API_KEY
)

SYSTEM_PROMPT = """You are an expert software engineer and AI assistant.
Think through problems step by step inside <thinking>...</thinking> tags,
then give your final answer. For tool calls, output valid JSON in the correct format.
Be thorough, correct, and explain your reasoning clearly."""

# Tool definitions (Qwen3 native format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a bash command in the terminal",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to run"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }
        }
    },
]

def load_prompts(filename):
    """Load prompts from a text file (one per line)."""
    path = Path(f"data/prompts/{filename}")
    if not path.exists():
        print(f"Warning: {path} not found")
        return []

    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def generate_example(prompt, use_tools=False):
    """Call Kimi K2 to generate a single example."""
    try:
        kwargs = {
            "model": "moonshot-v1",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        if use_tools:
            kwargs["tools"] = TOOLS

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message

    except Exception as e:
        print(f"Error generating example: {e}")
        return None

def to_jsonl_entry(user_prompt, assistant_msg):
    """Convert assistant message to JSONL format for training."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Handle tool calls
    if hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
        tool_calls_list = []
        for tc in assistant_msg.tool_calls:
            tool_calls_list.append({
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            })

        messages.append({
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": tool_calls_list
        })
    else:
        # Regular text response
        messages.append({
            "role": "assistant",
            "content": assistant_msg.content
        })

    return {"messages": messages}

def generate_dataset():
    """Generate training dataset from Kimi K2."""

    # Create output directory
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    output_file = Path("data/raw/kimi_examples.jsonl")

    # Load all prompts
    print("Loading prompts...")
    coding_prompts = load_prompts("kimi_coding.txt")
    debug_prompts = load_prompts("kimi_debugging.txt")
    tool_prompts = load_prompts("kimi_tools.txt")
    agent_prompts = load_prompts("kimi_agents.txt")

    # Distribution of examples
    examples_to_generate = [
        ("Coding", coding_prompts[:2000], False),
        ("Debugging", debug_prompts[:1500], False),
        ("Tool Use", tool_prompts[:2000], True),  # Use tools enabled
        ("Agent Tasks", agent_prompts[:1500], True),  # Use tools enabled
    ]

    total_to_generate = sum(len(prompts) for _, prompts, _ in examples_to_generate)
    print(f"Generating {total_to_generate} examples from Kimi K2...")

    generated_count = 0

    with open(output_file, "w") as f:
        for category, prompts, use_tools in examples_to_generate:
            print(f"\n{category} ({len(prompts)} examples):")

            for prompt in tqdm(prompts, desc=category):
                msg = generate_example(prompt, use_tools=use_tools)

                if msg and msg.content:
                    entry = to_jsonl_entry(prompt, msg)
                    f.write(json.dumps(entry) + "\n")
                    generated_count += 1

                # Rate limiting (Kimi has rate limits)
                time.sleep(0.5)

    print(f"\n✅ Generated {generated_count} examples → {output_file}")
    return generated_count

def split_train_val():
    """Split raw data into train/val (90/10)."""
    raw_file = Path("data/raw/kimi_examples.jsonl")
    train_file = Path("data/train.jsonl")
    val_file = Path("data/val.jsonl")

    # Load all examples
    examples = []
    with open(raw_file) as f:
        for line in f:
            examples.append(json.loads(line))

    # Shuffle and split
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)

    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    # Write train
    with open(train_file, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    # Write val
    with open(val_file, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n✅ Split data:")
    print(f"   Train: {len(train_examples)} examples → {train_file}")
    print(f"   Val: {len(val_examples)} examples → {val_file}")

if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3-8B Distillation: Generate Data from Kimi K2")
    print("=" * 60)

    # Generate data
    count = generate_dataset()

    if count > 0:
        # Split into train/val
        print("\nSplitting into train/val sets...")
        split_train_val()
        print("\n✅ Data generation complete! Ready for Phase 3 training.")
    else:
        print("\n❌ No examples generated. Check API key and rate limits.")
