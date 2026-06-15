"""
evaluation/evaluate.py
MT-LDI-MDS — Evaluation & Ablation Experiments

Loads trained checkpoints, evaluates on the held-out test set, and runs
three comparison experiments:
  1. Single-teacher baseline (only kd_loss_A)
  2. Multi-teacher fixed weights
  3. Multi-teacher learned weights  ← main contribution

Saves per-class classification report, confusion matrix, inference latency,
and a summary table to evaluation/results.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from utils.paths import get_project_paths

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
paths = get_project_paths(__file__)
SCRIPT_DIR = paths["EVALUATION_DIR"]
PROJECT_ROOT = paths["PROJECT_ROOT"]
DATA_DIR = paths["DATA_DIR"]
TRAINING_DIR = paths["TRAINING_DIR"]
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.json")

sys.path.insert(0, PROJECT_ROOT)
from student.bilstm_student import BiLSTMStudent
from aggregator.weighted_aggregator import WeightedAggregator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTACK_MAP = {
    "DoS": 1, "DoSDisruptive": 1, "DoSRandom": 1,
    "DoSRandomSybil": 1, "DoSDisruptiveSybil": 1,
    "Disruptive": 1, "DelayedMessages": 1,
    "GridSybil": 2,
    "ConstPosOffset": 3, "ConstPos": 3,
    "RandomPosOffset": 4, "RandomPos": 4,
    "EventualStop": 5,
    "ConstSpeedOffset": 6, "ConstSpeed": 6,
    "RandomSpeedOffset": 7, "RandomSpeed": 7,
    "DataReplaySybil": 8, "DataReplay": 8,
}

LABEL_NAMES = {
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

NUM_CLASSES = 9
FEATURE_COLS = [
    "type", "rcvTime",
    "pos_0", "pos_1", "pos_noise_0", "pos_noise_1",
    "spd_0", "spd_1", "spd_noise_0", "spd_noise_1",
    "acl_0", "acl_1", "acl_noise_0", "acl_noise_1",
    "hed_0", "hed_1", "hed_noise_0", "hed_noise_1",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_test_dataset():
    """Load full VeReMi, return test split (same 70/15/15 as training)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    csv_path = os.path.join(DATA_DIR, "veremi.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Please run data/split_dataset.py first."
        )
    df = pd.read_csv(csv_path, low_memory=False)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["numeric_label"] = df["attack_type"].map(ATTACK_MAP).fillna(0).astype(int)

    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["numeric_label"].to_numpy(dtype=np.int64)

    # Reproduce 70 / 15 / 15 split with same random_state
    _, X_tmp, _, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    _, X_test, _, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    return X_test, y_test


def build_test_loader(X, y, batch_size=32):
    # Unsqueeze to add seq_len=1
    X_t = torch.from_numpy(X).unsqueeze(1)
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


class TeacherAligner(nn.Module):
    """Same as in train_multiteacher.py."""
    def __init__(self, teacher_dim, student_dim=128):
        super().__init__()
        self.linear = nn.Linear(teacher_dim, student_dim)

    def forward(self, x):
        return self.linear(x)


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------
def evaluate_checkpoint(checkpoint_path, device, test_loader):
    """
    Load a checkpoint and evaluate on the test set.
    Returns a dict with metrics and inference latency per sample (ms).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    input_dim = ckpt["student_input_dim"]
    teacher_dim = ckpt["teacher_dim"]
    mode = ckpt.get("aggregator_mode", "learnable")

    student = BiLSTMStudent(input_dim=input_dim, num_classes=NUM_CLASSES).to(device)
    aggregator = WeightedAggregator(hidden_dim=teacher_dim, mode=mode, student_hidden_dim=256).to(device)

    student.load_state_dict(ckpt["student_state_dict"])
    aggregator.load_state_dict(ckpt["aggregator_state_dict"])
    student.eval()
    aggregator.eval()

    # Optionally load aligners (not needed for pure inference, but kept for completeness)
    aligners = []
    for sd in ckpt.get("aligners_state_dict", []):
        if sd is not None:
            al = TeacherAligner(teacher_dim, 128).to(device)
            al.load_state_dict(sd)
            al.eval()
            aligners.append(al)
        else:
            aligners.append(None)

    all_preds = []
    all_labels = []
    all_probs = []

    # Measure inference latency
    start = time.time()
    with torch.no_grad():
        for x, y_batch in test_loader:
            x = x.to(device)
            logits = student(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
            all_probs.append(probs.cpu().numpy())
    end = time.time()

    total_samples = len(all_labels)
    avg_ms_per_sample = ((end - start) / total_samples) * 1000.0

    # Classification report
    target_names = [LABEL_NAMES.get(i, f"class_{i}") for i in range(NUM_CLASSES)]
    report = classification_report(
        all_labels, all_preds, target_names=target_names,
        output_dict=True, zero_division=0
    )

    # Macro-average from report
    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]
    accuracy = report["accuracy"]

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    # λ weights
    lambda_weights = aggregator.get_weights()

    results = {
        "checkpoint": checkpoint_path,
        "mode": mode,
        "accuracy": float(accuracy),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "avg_inference_ms": float(avg_ms_per_sample),
        "lambda_weights": lambda_weights,
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in target_names
        },
        "confusion_matrix": cm.tolist(),
        "num_samples": total_samples,
    }

    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision(m):  {macro_precision:.4f}")
    print(f"Recall(m):     {macro_recall:.4f}")
    print(f"F1(m):         {macro_f1:.4f}")
    print(f"Latency:       {avg_ms_per_sample:.4f} ms/sample")
    return results


# ---------------------------------------------------------------------------
# Baseline training helpers
# ---------------------------------------------------------------------------
def ensure_trained(checkpoint_path, mode, kd_teachers, auto_train=False):
    """If checkpoint missing, either raise or trigger training via subprocess."""
    if os.path.exists(checkpoint_path):
        return
    msg = f"Checkpoint missing: {checkpoint_path}"
    if not auto_train:
        raise FileNotFoundError(
            f"{msg}\n"
            f"Train it with:\n"
            f"  python training/train_multiteacher.py --mode {mode} --kd-teachers {kd_teachers} --checkpoint {checkpoint_path}\n"
            f"Or run this script with --auto-train to build missing baselines automatically."
        )
    print(f"[INFO] {msg} — auto-training now (mode={mode}, kd-teachers={kd_teachers}) ...")
    cmd = [
        sys.executable, os.path.join(TRAINING_DIR, "train_multiteacher.py"),
        "--mode", mode,
        "--kd-teachers", kd_teachers,
        "--checkpoint", checkpoint_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"[INFO] Training complete: {checkpoint_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MT-LDI-MDS evaluation & comparison.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu.")
    parser.add_argument("--auto-train", action="store_true",
                        help="Automatically train missing baseline checkpoints.")
    parser.add_argument("--single-ckpt", type=str,
                        default=os.path.join(TRAINING_DIR, "best_model_single.pt"),
                        help="Checkpoint for single-teacher baseline.")
    parser.add_argument("--fixed-ckpt", type=str,
                        default=os.path.join(TRAINING_DIR, "best_model_fixed.pt"),
                        help="Checkpoint for fixed-weight baseline.")
    parser.add_argument("--learned-ckpt", type=str,
                        default=os.path.join(TRAINING_DIR, "best_model.pt"),
                        help="Checkpoint for learned-weight main model.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device = torch.device("cpu")

    print("[INFO] Loading test dataset ...")
    X_test, y_test = load_test_dataset()
    test_loader = build_test_loader(X_test, y_test, batch_size=args.batch_size)
    print(f"[INFO] Test samples: {len(y_test)}")

    # ------------------------------------------------------------------
    # Ensure all checkpoints exist (or raise / train)
    # ------------------------------------------------------------------
    ensure_trained(args.single_ckpt, mode="learnable", kd_teachers="A", auto_train=args.auto_train)
    ensure_trained(args.fixed_ckpt, mode="fixed", kd_teachers="ABC", auto_train=args.auto_train)
    ensure_trained(args.learned_ckpt, mode="learnable", kd_teachers="ABC", auto_train=args.auto_train)

    # ------------------------------------------------------------------
    # Experiment 1: Single-teacher baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Single-Teacher Baseline (Teacher A only)")
    print("=" * 60)
    res_single = evaluate_checkpoint(args.single_ckpt, device, test_loader)

    # ------------------------------------------------------------------
    # Experiment 2: Fixed weights
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Multi-Teacher Fixed Weights")
    print("=" * 60)
    res_fixed = evaluate_checkpoint(args.fixed_ckpt, device, test_loader)

    # ------------------------------------------------------------------
    # Experiment 3: Learned weights (main contribution)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Multi-Teacher Learned Weights (Main)")
    print("=" * 60)
    res_learned = evaluate_checkpoint(args.learned_ckpt, device, test_loader)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY TABLE")
    print("=" * 60)
    header = f"{'Experiment':<35} {'Acc':>8} {'Prec(m)':>9} {'Rec(m)':>9} {'F1(m)':>9} {'Latency':>10}"
    print(header)
    print("-" * len(header))

    rows = [
        ("1. Single-Teacher (A only)", res_single),
        ("2. Multi-Teacher Fixed Weights", res_fixed),
        ("3. Multi-Teacher Learned Weights", res_learned),
    ]

    for name, r in rows:
        print(
            f"{name:<35} "
            f"{r['accuracy']:>8.4f} "
            f"{r['precision_macro']:>9.4f} "
            f"{r['recall_macro']:>9.4f} "
            f"{r['f1_macro']:>9.4f} "
            f"{r['avg_inference_ms']:>9.2f}ms"
        )

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiments": {
                    "single_teacher": res_single,
                    "fixed_weights": res_fixed,
                    "learned_weights": res_learned,
                }
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n[INFO] Results saved → {RESULTS_PATH}")
    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()