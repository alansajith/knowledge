"""
teachers/finetune_teacher.py
MT-LDI-MDS — Teacher LLM Fine-Tuning & Embedding Extraction

This script:
1. Loads the appropriate VeReMi split (A/B/C).
2. Builds an instruction-tuning JSONL dataset.
3. Fine-tunes Qwen3-8B with LoRA using PEFT + SFTTrainer on NVIDIA GPU/CUDA.
4. After fine-tuning, extracts mean-pooled hidden-state embeddings from the
   base Qwen3-8B model using PyTorch/CUDA and saves them as .npy arrays.

Targets NVIDIA GPU cluster (Google Cloud / University) — PyTorch CUDA, PEFT for LoRA.
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from utils.paths import get_project_paths

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
paths = get_project_paths(__file__)
DATA_DIR = paths["DATA_DIR"]
SCRIPT_DIR = paths["TEACHERS_DIR"]
PROJECT_ROOT = paths["PROJECT_ROOT"]

TEACHER_SPLITS = {"A": "split_A.csv", "B": "split_B.csv", "C": "split_C.csv"}

# Device setup: prefer CUDA if available, fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model identifier
HF_MODEL_NAME = "Qwen/Qwen3-8B"

# Number of raw BSM features expected in the CSV (excluding metadata)
FEATURE_COLS = [
    "type", "rcvTime",
    "pos_0", "pos_1", "pos_noise_0", "pos_noise_1",
    "spd_0", "spd_1", "spd_noise_0", "spd_noise_1",
    "acl_0", "acl_1", "acl_noise_0", "acl_noise_1",
    "hed_0", "hed_1", "hed_noise_0", "hed_noise_1",
]

NUMERIC_LABEL_TO_NAME = {
    0: "benign",
    1: "DoS",
    2: "Sybil",
    3: "fixed_position",
    4: "random_position",
    5: "eventual_stop",
    6: "fixed_speed",
    7: "random_speed",
    8: "data_replay",
}


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------
def build_instruction_prompt(row: pd.Series) -> str:
    """Format a single row into the instruction-tuning input prompt."""
    # Only include columns that actually exist (defensive)
    feature_vals = [str(row.get(c, 0.0)) for c in FEATURE_COLS if c in row.index]
    features_str = ", ".join(feature_vals)
    variability = row.get("nn_variability", 0.0)
    lbl = int(row["numeric_label"])
    attack_name = NUMERIC_LABEL_TO_NAME.get(lbl, "unknown")

    prompt = (
        f"Given this vehicular BSM log sample: {features_str}, and its "
        f"inter-sample variability from nearest neighbor: {variability:.6f}, "
        f"generate a new realistic synthetic vehicular log for attack type: {attack_name}"
    )
    return prompt


def build_synthetic_output(row: pd.Series) -> str:
    """
    Generate a synthetic BSM log string in the same feature format.
    We add a small amount of Gaussian noise to the original features so the
    output is a *new* realistic sample rather than a verbatim copy.
    """
    feature_vals = [float(row.get(c, 0.0)) for c in FEATURE_COLS if c in row.index]
    # Small perturbation (std = 1% of absolute value, clamped)
    perturbed = []
    for v in feature_vals:
        noise = np.random.normal(0, max(abs(v) * 0.01, 0.001))
        perturbed.append(f"{v + noise:.6f}")
    return ", ".join(perturbed)


def generate_jsonl_dataset(df: pd.DataFrame, out_path: str):
    """Write the teacher split as instruction-tuning JSONL (chat format)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building JSONL"):
            prompt = build_instruction_prompt(row)
            completion = build_synthetic_output(row)
            # HF transformers auto-detects chat format via "messages" key
            example = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"JSONL dataset written: {out_path} ({len(df)} examples)")


# ---------------------------------------------------------------------------
# PEFT LoRA fine-tuning (CUDA)
# ---------------------------------------------------------------------------
def finetune_with_peft(data_dir: str, adapter_path: str, device: torch.device):
    """Fine-tune Qwen3-8B with LoRA using PEFT + SFTTrainer on CUDA/CPU."""
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from trl import SFTTrainer

    print("\n=== Loading base model Qwen/Qwen3-8B (4-bit for GPU efficiency) ===")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["self_attn.q_proj", "self_attn.v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load JSONL datasets
    train_ds = load_dataset("json", data_files=os.path.join(data_dir, "train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=os.path.join(data_dir, "valid.jsonl"), split="train")

    # Convert messages to text strings for SFTTrainer
    def format_messages(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {"text": text}

    train_ds = train_ds.map(format_messages)
    val_ds = val_ds.map(format_messages)

    training_args = TrainingArguments(
        output_dir=os.path.join(adapter_path, "training_output"),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=100,
        save_strategy="steps",
        save_steps=max(len(train_ds) // 10, 100),
        eval_strategy="steps",
        eval_steps=max(len(train_ds) // 5, 200),
        load_best_model_at_end=True,
        bf16=False,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
    )

    print("=== Starting LoRA training ===")
    trainer.train()

    print(f"=== Saving adapter to {adapter_path} ===")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"[INFO] LoRA fine-tuning complete. Adapters saved to {adapter_path}")


# ---------------------------------------------------------------------------
# Embedding extraction (PyTorch / HF / CUDA)
# ---------------------------------------------------------------------------
def extract_embeddings(df: pd.DataFrame, teacher: str, batch_size: int = 16):
    """
    Load Qwen/Qwen3-8B via HF transformers, run all training prompts through
    the model, mean-pool the last hidden state, save as .npy.
    """
    print(f"\nLoading HF model {HF_MODEL_NAME} for embedding extraction ...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        HF_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=False,
    )
    model.eval()
    # If device_map is None (CPU), move manually
    if DEVICE.type == "cpu":
        model = model.to(DEVICE)

    embeddings_list = []
    prompts = [build_instruction_prompt(row) for _, row in df.iterrows()]

    print(f"Extracting embeddings for {len(prompts)} samples on {DEVICE} ...")
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Embedding batches"):
            batch_prompts = prompts[i : i + batch_size]

            # Apply Qwen3 chat template without generation prefix / thinking tokens
            texts = []
            for p in batch_prompts:
                messages = [{"role": "user", "content": p}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                texts.append(text)

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)

            # Mean-pool over real (non-padding) tokens
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_emb = torch.sum(last_hidden * attention_mask, dim=1)
            mean_emb = sum_emb / attention_mask.sum(dim=1).clamp(min=1e-9)

            embeddings_list.append(mean_emb.cpu().to(torch.float32).numpy())

    all_embeddings = np.concatenate(embeddings_list, axis=0)
    emb_path = os.path.join(SCRIPT_DIR, f"embeddings_{teacher}.npy")
    np.save(emb_path, all_embeddings)
    print(f"Saved embeddings -> {emb_path} (shape={all_embeddings.shape})")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Qwen3-8B teacher on a VeReMi split and extract embeddings."
    )
    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        choices=["A", "B", "C"],
        help="Which teacher subset to fine-tune (A, B, or C).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for embedding extraction (default: cuda).",
    )
    parser.add_argument(
        "--skip-lora",
        action="store_true",
        help="Skip LoRA fine-tuning and only run embedding extraction (for debug).",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding extraction after LoRA (for debug).",
    )
    parser.add_argument(
        "--emb-batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction (default: 16).",
    )
    args = parser.parse_args()

    teacher = args.teacher
    split_file = os.path.join(DATA_DIR, TEACHER_SPLITS[teacher])
    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            "Run data/split_dataset.py first to generate the splits."
        )

    df = pd.read_csv(split_file)
    print(f"Loaded split '{teacher}': {len(df)} rows")

    # ------------------------------------------------------------------
    # 1) JSONL data generation
    # ------------------------------------------------------------------
    data_out_dir = os.path.join(SCRIPT_DIR, f"teacher_{teacher}_data")
    os.makedirs(data_out_dir, exist_ok=True)
    jsonl_path = os.path.join(data_out_dir, "train.jsonl")

    # For validation we hold out 10% of the split
    val_df = df.sample(frac=0.1, random_state=42)
    train_df = df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    generate_jsonl_dataset(train_df, jsonl_path)
    generate_jsonl_dataset(val_df, os.path.join(data_out_dir, "valid.jsonl"))

    # ------------------------------------------------------------------
    # 2) Adapter path calculation
    # ------------------------------------------------------------------
    adapter_path = os.path.join(SCRIPT_DIR, f"teacher_{teacher}_adapters")

    # ------------------------------------------------------------------
    # 3) PEFT LoRA fine-tuning
    # ------------------------------------------------------------------
    if not args.skip_lora:
        print(f"\n[INFO] Starting PEFT LoRA fine-tuning for Teacher {teacher} ...")
        finetune_with_peft(data_out_dir, adapter_path, DEVICE)
    else:
        print("[INFO] Skipping LoRA fine-tuning (--skip-lora).")

    # ------------------------------------------------------------------
    # 4) Embedding extraction on the full split (train+val for KD)
    # ------------------------------------------------------------------
    if not args.skip_embeddings:
        extract_embeddings(df, teacher=teacher, batch_size=args.emb_batch_size)
    else:
        print("[INFO] Skipping embedding extraction (--skip-embeddings).")

    print(f"\nTeacher {teacher} pipeline complete.")


if __name__ == "__main__":
    main()