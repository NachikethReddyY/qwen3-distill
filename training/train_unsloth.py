#!/usr/bin/env python3
"""
QLoRA fine-tuning of Qwen3-8B using Unsloth with MLX.
Trains on Apple Silicon, exports to GGUF format.
"""

import json
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading Qwen3-8B model with MLX optimization...")
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-8B",
    max_seq_length=8192,
    load_in_4bit=True,
)

if tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained("./models/qwen3-8b", trust_remote_code=True)

print("Preparing model for QLoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Load data
def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

print("Loading training data...")
train_data = load_jsonl("data/train.jsonl")
val_data   = load_jsonl("data/val.jsonl")
# Also include dataset 2 if available
import os as _os
if _os.path.exists("data/train2.jsonl"):
    train2 = load_jsonl("data/train2.jsonl")
    val2   = load_jsonl("data/val2.jsonl") if _os.path.exists("data/val2.jsonl") else []
    train_data += train2
    val_data   += val2
    print(f"✓ Merged dataset2: +{len(train2)} train, +{len(val2)} val")
print(f"✓ Total {len(train_data)} training examples")
print(f"✓ Total {len(val_data)} validation examples")

# Format messages
def format_messages(messages):
    text = ""
    for msg in messages:
        text += f"{msg['role']}: {msg['content']}\n"
    return text.strip()

train_texts = [format_messages(d["messages"]) for d in train_data]

# Simple dataset wrapper
class SimpleDataset:
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx]}

train_dataset = SimpleDataset(train_texts)

print("\n" + "="*60)
print("Setting up training with Unsloth...")
print("="*60)

# Training configuration
training_args = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    num_train_epochs=2,
    learning_rate=2e-5,
    fp16=False,  # MLX handles this
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    report_to=[],
    seed=42,
)

# Use native SFTTrainer with minimal args
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=8192,
    args=training_args,
    packing=False,
)

print("\n" + "="*60)
print("Starting training...")
print("="*60 + "\n")

# Train
trainer.train()

# Save adapter
print("\n" + "="*60)
print("Saving adapter to PyTorch format...")
model.save_pretrained("./adapters/qwen3-distill")
tokenizer.save_pretrained("./adapters/qwen3-distill")
print("✅ Adapter saved to ./adapters/qwen3-distill/")

# Export to GGUF
print("\n" + "="*60)
print("Exporting to GGUF format...")
print("="*60)

try:
    # Merge adapter with base model for export
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained("./adapters/qwen3-distill-merged")
    tokenizer.save_pretrained("./adapters/qwen3-distill-merged")

    print("✅ Merged model saved to ./adapters/qwen3-distill-merged/")
    print("\nTo convert to GGUF, use llama.cpp:")
    print("  python llama.cpp/convert.py ./adapters/qwen3-distill-merged/ --outtype q4_0 -o qwen3-distill.gguf")

except Exception as e:
    print(f"Note: Could not auto-export to GGUF: {e}")
    print("To export manually:")
    print("  1. pip install gguf")
    print("  2. python convert_to_gguf.py ./adapters/qwen3-distill-merged/")

print("="*60)
