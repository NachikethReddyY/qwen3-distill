#!/usr/bin/env python3
"""
QLoRA fine-tuning of Qwen3-8B on generated training data using Unsloth.
Runs on Nosana RTX 3090 or local GPU.
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Load base model with 4-bit quantization for efficiency
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-8B",
    max_seq_length=8192,
    load_in_4bit=True,
)

# Prepare model for QLoRA fine-tuning
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

# Load training and validation datasets
dataset = load_dataset("json", data_files={
    "train": "data/train.jsonl",
    "validation": "data/val.jsonl"
})

# Configure training arguments
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="messages",
    args=TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        fp16=True,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
    ),
)

# Train the model
print("Starting training...")
trainer.train()

# Save the LoRA adapter
print("Saving adapter...")
model.save_pretrained("./adapters/qwen3-distill")
tokenizer.save_pretrained("./adapters/qwen3-distill")
print("✅ Training complete! Adapter saved to ./adapters/qwen3-distill/")
