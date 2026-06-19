#!/usr/bin/env python3
"""
PyTorch QLoRA training for Qwen3.5-9B on GPU (CUDA/Nosana)
Same dataset as MLX version, but using transformers + bitsandbytes for GPU acceleration.

Usage:
  python training/train_nosana.py

Nosana deployment:
  nosana jobs submit --image qwen3-distill:latest \
    --gpu nvidia-a100 --command "python training/train_nosana.py"
"""

import json
import torch
import argparse
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3.5-9B"  # Base model (not quantized)
ADAPTER_OUTPUT = "./adapters/qwen35-9b-v3-gpu"
DATA_DIR = "data/mlx_data_combined"

LEARNING_RATE = 2e-5
BATCH_SIZE = 4  # GPU can handle larger batches than M4
GRAD_ACCUM = 2
ITERS = 1600  # ~5 epochs (1200 = 4 epochs, 2800+ = overfitting)
MAX_SEQ_LENGTH = 1024

# ── Helper Functions ─────────────────────────────────────────────
def load_jsonl(path):
    """Load JSONL dataset from path."""
    if not Path(path).exists():
        logger.warning(f"File not found: {path}")
        return []
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON: {e}")
    return data

def format_chat_for_training(example):
    """Convert chat format to training text."""
    messages = example.get("messages", [])
    text_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            text_parts.append(f"[SYSTEM]\n{content}\n[/SYSTEM]\n")
        elif role == "user":
            text_parts.append(f"[USER]\n{content}\n[/USER]\n")
        elif role == "assistant":
            text_parts.append(f"[ASSISTANT]\n{content}\n[/ASSISTANT]\n")
    return "".join(text_parts)

def setup_model_and_tokenizer():
    """Initialize model with 4-bit quantization for GPU."""
    logger.info(f"Loading {MODEL_ID} with 4-bit quantization...")

    # 4-bit quantization config for CUDA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Requires flash-attn
    )

    # Ensure gradient checkpointing
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded. Params: {model.num_parameters() / 1e9:.2f}B")
    return model, tokenizer

def setup_lora(model):
    """Configure LoRA for fine-tuning."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def main(args):
    # Load and prepare data
    logger.info("Loading datasets...")
    train_data = load_jsonl(f"{DATA_DIR}/train.jsonl")
    val_data = load_jsonl(f"{DATA_DIR}/valid.jsonl")

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    if not train_data:
        raise ValueError("No training data found!")

    # Format data for SFT
    train_data = [{"text": format_chat_for_training(ex)} for ex in train_data]
    val_data = [{"text": format_chat_for_training(ex)} for ex in val_data]

    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    model = setup_lora(model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=ADAPTER_OUTPUT,
        num_train_epochs=round(ITERS / len(train_data), 1) if train_data else 1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=min(400, len(train_data) // BATCH_SIZE),
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        bf16=True,  # Use bfloat16 on modern GPUs
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to=["tensorboard"],
        seed=42,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,  # Set to True for longer sequences if memory allows
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving adapter to {ADAPTER_OUTPUT}...")
    model.save_pretrained(ADAPTER_OUTPUT)
    tokenizer.save_pretrained(ADAPTER_OUTPUT)

    logger.info("✅ Training complete!")
    logger.info(f"   Adapter saved to: {ADAPTER_OUTPUT}")
    logger.info("   Next: merge adapter with base model for inference")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--output-dir", type=str, default=ADAPTER_OUTPUT)
    args = parser.parse_args()

    main(args)
