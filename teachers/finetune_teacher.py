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

# ---------------------------------------------------------------------------
# A100 40GB Optimizations
# ---------------------------------------------------------------------------
# Enable TF32 for faster matmul on Ampere GPUs (A100)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Enable cuDNN benchmark for optimal convolution algorithms
torch.backends.cudnn.benchmark = True

# Device setup: prefer CUDA if available, fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mixed precision settings for A100 (BF16 preferred on Ampere)
USE_AMP = DEVICE.type == "cuda"
AMP_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

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
def finetune_with_peft(data_dir: str, adapter_path: str, device: torch.device, grad_accum_steps: int = 4, use_compile: bool = False):
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

    from transformers import AutoConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use BF16 for A100
        bnb_4bit_use_double_quant=True,
    )

    from transformers import AutoConfig

    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Build a balanced multi-GPU device map — GPU 0 has embed_tokens so give it fewer layers
    num_gpus = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(HF_MODEL_NAME, trust_remote_code=False)
    num_layers = config.num_hidden_layers  # e.g. 32 for Qwen3-8B

    # Dynamic split: distribute layers evenly, but GPU 0 gets embed_tokens so give it 2 fewer layers
    layers_per_gpu = num_layers // num_gpus
    device_map = {"model.embed_tokens": 0}
    for idx in range(num_layers):
        if idx < layers_per_gpu - 2:
            gpu_id = 0
        else:
            gpu_id = min((idx - (layers_per_gpu - 2)) // layers_per_gpu + 1, num_gpus - 1)
        device_map[f"model.layers.{idx}"] = gpu_id
    device_map["model.norm"] = num_gpus - 1
    device_map["lm_head"] = num_gpus - 1

    print(f"Device map: {device_map}")

    # Enable Flash Attention 2 for A100 (requires flash-attn package)
    # Flash Attention 2 provides significant speedup on Ampere GPUs
    try:
        import flash_attn
        use_flash_attn = True
        print("[INFO] Flash Attention 2 available - enabling for A100")
    except ImportError:
        use_flash_attn = False
        print("[WARN] Flash Attention 2 not installed - using standard attention")

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2" if use_flash_attn else "eager",
        use_cache=False,  # Disable KV cache for training with gradient checkpointing
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # torch.compile for faster training on A100 (PyTorch 2.0+)
    if use_compile and device.type == "cuda":
        print("[INFO] Compiling model with torch.compile...")
        model = torch.compile(model, mode="max-autotune", fullgraph=False)

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
        per_device_train_batch_size=8,  # Increased for A100 40GB
        gradient_accumulation_steps=grad_accum_steps,  # Configurable gradient accumulation
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_steps=max(len(train_ds) // 10 * 3 // 100, 10),
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="steps",
        save_steps=max(len(train_ds) // 10, 100),
        eval_strategy="steps",
        eval_steps=max(len(train_ds) // 10, 100),
        load_best_model_at_end=True,
        bf16=True,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        # A100 optimizations
        tf32=True,  # Enable TF32 for Ampere GPUs
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
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
def extract_embeddings(df: pd.DataFrame, teacher: str, batch_size: int = 32):
    """
    Load Qwen/Qwen3-8B via HF transformers, run all training prompts through
    the model, mean-pool the last hidden state, save as .npy.
    """
    print(f"\nLoading HF model {HF_MODEL_NAME} for embedding extraction ...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=False)
    
    # Try to enable Flash Attention 2 for faster inference
    try:
        import flash_attn
        use_flash_attn = True
        print("[INFO] Flash Attention 2 available - enabling for embedding extraction")
    except ImportError:
        use_flash_attn = False
        print("[WARN] Flash Attention 2 not installed - using standard attention")
    
    model = AutoModel.from_pretrained(
        HF_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=False,
        attn_implementation="flash_attention_2" if use_flash_attn else "eager",
    )
    model.eval()
    # If device_map is None (CPU), move manually
    if DEVICE.type == "cpu":
        model = model.to(DEVICE)
    
    # torch.compile for faster inference on A100
    if DEVICE.type == "cuda":
        model = torch.compile(model, mode="max-autotune", fullgraph=False)

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

            # Use mixed precision (BF16) for faster inference on A100
            with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
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
        default=32,
        help="Batch size for embedding extraction (default: 32 for A100).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps for LoRA training (default: 4).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for faster training (requires PyTorch 2.0+).",
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