#!/usr/bin/env python3
"""
Helper script to easily add new training examples.
Run this to interactively add examples or modify existing ones.
"""

import json
from pathlib import Path

def add_example_interactive():
    """Add a new example via interactive prompts."""
    print("\n" + "="*60)
    print("ADD NEW TRAINING EXAMPLE")
    print("="*60)

    categories = ["react", "javascript", "python", "coding", "reasoning", "debugging", "tools", "agents"]
    print(f"\nAvailable categories: {', '.join(categories)}")
    category = input("Choose category: ").strip().lower()

    if category not in categories:
        print(f"❌ Invalid category. Must be one of: {', '.join(categories)}")
        return

    print("\nEnter the USER PROMPT (what the user asks):")
    print("(Press Enter twice when done)")
    lines = []
    while True:
        line = input()
        if not line:
            if lines and lines[-1] == "":
                lines.pop()
                break
            lines.append("")
        else:
            lines.append(line)
    prompt = "\n".join(lines)

    if not prompt:
        print("❌ Prompt cannot be empty")
        return

    print("\nEnter the ASSISTANT ANSWER:")
    print("Include:")
    print("  1. <thinking>...</thinking> section")
    print("  2. Code block (```jsx, ```python, ```javascript)")
    print("  3. Explanation")
    print("(Press Enter twice when done)")
    lines = []
    while True:
        line = input()
        if not line:
            if lines and lines[-1] == "":
                lines.pop()
                break
            lines.append("")
        else:
            lines.append(line)
    answer = "\n".join(lines)

    if not answer:
        print("❌ Answer cannot be empty")
        return

    # Create training entry
    entry = {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert software engineer with deep knowledge of React, JavaScript, Python, animations, and data visualization.\nThink through problems step by step inside <thinking>...</thinking> tags, explaining your reasoning and approach.\nThen give your final answer with clean, well-structured code and detailed explanations.\nBe thorough, correct, and explain complex concepts clearly."
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
    }

    # Append to raw file
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / f"{category}_examples.jsonl"

    with open(raw_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\n✅ Example added to {raw_file}")
    print(f"📊 Total examples in {category}: {sum(1 for _ in open(raw_file))}")

    # Ask if user wants to regenerate
    regenerate = input("\nRegenerate train/val splits? (y/n): ").strip().lower()
    if regenerate == "y":
        import subprocess
        subprocess.run(["python", "merge_data.py"])

def view_stats():
    """Show statistics about existing examples."""
    print("\n" + "="*60)
    print("TRAINING DATA STATISTICS")
    print("="*60)

    raw_dir = Path("data/raw")
    total = 0

    for jsonl_file in sorted(raw_dir.glob("*_examples.jsonl")):
        count = sum(1 for _ in open(jsonl_file))
        category = jsonl_file.stem.replace("_examples", "")
        print(f"  {category:15} {count:3} examples")
        total += count

    print(f"\n  {'TOTAL':15} {total:3} examples")

    # Check merged files
    train_file = Path("data/train.jsonl")
    val_file = Path("data/val.jsonl")

    if train_file.exists() and val_file.exists():
        train_count = sum(1 for _ in open(train_file))
        val_count = sum(1 for _ in open(val_file))
        print(f"\n  Train split:     {train_count} examples (90%)")
        print(f"  Val split:       {val_count} examples (10%)")

def show_menu():
    """Show main menu."""
    print("\n" + "="*60)
    print("TRAINING DATA HELPER")
    print("="*60)
    print("1. Add new example (interactive)")
    print("2. View statistics")
    print("3. Regenerate train/val splits (merge_data.py)")
    print("4. Exit")
    return input("\nChoose option (1-4): ").strip()

def main():
    while True:
        choice = show_menu()

        if choice == "1":
            add_example_interactive()
        elif choice == "2":
            view_stats()
        elif choice == "3":
            print("\nRunning merge_data.py...")
            import subprocess
            subprocess.run(["python", "merge_data.py"])
        elif choice == "4":
            print("✅ Done!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
