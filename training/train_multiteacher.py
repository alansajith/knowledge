"""
training/train_multiteacher.py
MT-LDI-MDS — Multi-Teacher Distillation Training Loop

Implements the core contribution:
  total_loss = ce_loss + λ1*kd_A + λ2*kd_B + λ3*kd_C

where λ are learnable softmax weights from the WeightedAggregator,
kd_* are MSE distances between the student's intermediate 128-dim feature
vector and each teacher's (projected) hidden-state embedding.

Train/val/test split: 70 / 15 / 15 stratified on the full VeReMi dataset.
Best model is saved to training/best_model.pt.
Training log is saved to training/training_log.csv.
"""

import argparse
import csv
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.paths import get_project_paths

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
paths = get_project_paths(__file__)
SCRIPT_DIR = paths["TRAINING_DIR"]
PROJECT_ROOT = paths["PROJECT_ROOT"]
DATA_DIR = paths["DATA_DIR"]
TEACHERS_DIR = paths["TEACHERS_DIR"]
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "best_model.pt")
LOG_PATH = os.path.join(SCRIPT_DIR, "training_log.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same mapping as split_dataset.py
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

NUM_CLASSES = 9
FEATURE_COLS = [
    "type", "rcvTime",
    "pos_0", "pos_1", "pos_noise_0", "pos_noise_1",
    "spd_0", "spd_1", "spd_noise_0", "spd_noise_1",
    "acl_0", "acl_1", "acl_noise_0", "acl_noise_1",
    "hed_0", "hed_1", "hed_noise_0", "hed_noise_1",
]


def load_dataset():
    """Load full VeReMi dataset and compute numeric labels."""
    csv_path = os.path.join(DATA_DIR, "veremi.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Please run data/split_dataset.py first or download the dataset from\n"
            "https://github.com/josephkamel/VeReMi-Dataset"
        )
    df = pd.read_csv(csv_path, low_memory=False)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["numeric_label"] = df["attack_type"].map(ATTACK_MAP).fillna(0).astype(int)

    # Select only numeric feature columns that actually exist
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["numeric_label"].to_numpy(dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Teacher embedding helpers
# ---------------------------------------------------------------------------
class TeacherEmbeddingDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset that optionally loads teacher embeddings
    via numpy memmap so multi-gigabyte .npy files do not fully reside in RAM.
    """

    def __init__(self, X, y, emb_A=None, emb_B=None, emb_C=None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        # Embeddings may be numpy arrays, memmaps, or None
        self.emb_A = emb_A
        self.emb_B = emb_B
        self.emb_C = emb_C
        self.has_kd = emb_A is not None or emb_B is not None or emb_C is not None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        label = self.y[idx]
        # Unsqueeze to add seq_len=1 for the student
        x = x.unsqueeze(0)  # (1, input_dim)
        if not self.has_kd:
            return x, label
        a = self._get_emb(self.emb_A, idx)
        b = self._get_emb(self.emb_B, idx)
        c = self._get_emb(self.emb_C, idx)
        return x, label, a, b, c

    @staticmethod
    def _get_emb(emb, idx):
        if emb is None:
            return torch.zeros(1)  # dummy placeholder
        val = emb[idx]
        if hasattr(val, "astype"):
            val = val.astype(np.float32)
        return torch.from_numpy(val).float()


def load_teacher_embeddings(train_len):
    """
    Load teacher embeddings from .npy files using memory mapping.
    Returns dict of raw numpy arrays (or None if missing / length mismatch).
    """
    embs = {}
    for t in ["A", "B", "C"]:
        path = os.path.join(TEACHERS_DIR, f"embeddings_{t}.npy")
        if os.path.exists(path):
            try:
                # memory-map so we don't pull the whole array into RAM
                arr = np.load(path, mmap_mode="r")
                if arr.shape[0] == train_len:
                    embs[t] = arr
                    print(f"[INFO] Loaded embeddings_{t}.npy → {arr.shape} (memmapped)")
                else:
                    print(
                        f"[WARN] embeddings_{t}.npy length {arr.shape[0]} != train_len {train_len}. "
                        f"Skipping KD for teacher {t}."
                    )
                    embs[t] = None
            except Exception as e:
                print(f"[WARN] Failed to load embeddings_{t}.npy: {e}. Skipping.")
                embs[t] = None
        else:
            print(f"[WARN] embeddings_{t}.npy not found. Skipping KD for teacher {t}.")
            embs[t] = None
    return embs


def collate_fn(batch):
    """Collate that handles optional teacher embedding fields."""
    if len(batch[0]) == 2:
        # No KD
        x = torch.stack([b[0] for b in batch])
        y = torch.stack([b[1] for b in batch])
        return x, y
    else:
        x = torch.stack([b[0] for b in batch])
        y = torch.stack([b[1] for b in batch])
        a = torch.stack([b[2] for b in batch])
        b_ = torch.stack([b[3] for b in batch])
        c = torch.stack([b[4] for b in batch])
        return x, y, a, b_, c


# ---------------------------------------------------------------------------
# Teacher alignment heads (project raw teacher embeddings → 128-dim student space)
# ---------------------------------------------------------------------------
class TeacherAligner(nn.Module):
    """Lightweight linear projector from teacher hidden dim to student hidden dim."""

    def __init__(self, teacher_dim: int, student_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(teacher_dim, student_dim)

    def forward(self, x):
        return self.linear(x)


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, aggregator, aligners, loader, criterion_ce, device, kd_teachers="ABC"):
    """Return (avg_loss, ce_loss, accuracy, f1_macro) on the given loader."""
    model.eval()
    aggregator.eval()
    for al in aligners:
        if al is not None:
            al.eval()

    total_loss = 0.0
    total_ce = 0.0
    all_preds = []
    all_labels = []
    count = 0

    has_kd = loader.dataset.has_kd
    use_a = "A" in kd_teachers
    use_b = "B" in kd_teachers
    use_c = "C" in kd_teachers

    for batch in loader:
        if has_kd:
            x, y, eA, eB, eC = batch
            eA = eA.to(device)
            eB = eB.to(device)
            eC = eC.to(device)
        else:
            x, y = batch
            eA = eB = eC = None

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        ce = criterion_ce(logits, y)
        loss = ce

        if has_kd:
            student_feat = model.get_intermediate_features()  # (B, 128)
            w = F.softmax(torch.stack([aggregator.lambda1, aggregator.lambda2, aggregator.lambda3]), dim=0)
            if use_a and aligners[0] is not None and eA is not None and eA.shape[-1] == aligners[0].linear.in_features:
                tA = aligners[0](eA)
                kd_A = F.mse_loss(student_feat, tA)
                loss = loss + w[0] * kd_A
            if use_b and aligners[1] is not None and eB is not None and eB.shape[-1] == aligners[1].linear.in_features:
                tB = aligners[1](eB)
                kd_B = F.mse_loss(student_feat, tB)
                loss = loss + w[1] * kd_B
            if use_c and aligners[2] is not None and eC is not None and eC.shape[-1] == aligners[2].linear.in_features:
                tC = aligners[2](eC)
                kd_C = F.mse_loss(student_feat, tC)
                loss = loss + w[2] * kd_C

        total_loss += loss.item()
        total_ce += ce.item()
        count += 1

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / max(count, 1)
    avg_ce = total_ce / max(count, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, avg_ce, acc, f1


def train_epoch(model, aggregator, aligners, loader, optimizer, criterion_ce, device, has_kd, kd_teachers="ABC"):
    """Run one training epoch and return dict of average losses."""
    model.train()
    aggregator.train()
    for al in aligners:
        if al is not None:
            al.train()

    use_a = "A" in kd_teachers
    use_b = "B" in kd_teachers
    use_c = "C" in kd_teachers

    total_loss = 0.0
    total_ce = 0.0
    total_kd_A = 0.0
    total_kd_B = 0.0
    total_kd_C = 0.0
    count = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        if has_kd:
            x, y, eA, eB, eC = batch
            eA = eA.to(device)
            eB = eB.to(device)
            eC = eC.to(device)
        else:
            x, y = batch
            eA = eB = eC = None

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        ce = criterion_ce(logits, y)
        loss = ce

        kd_A_val = kd_B_val = kd_C_val = torch.tensor(0.0, device=device)

        if has_kd:
            # Compute λ weights WITH gradient tracking so they learn
            raw_lambdas = torch.stack([aggregator.lambda1, aggregator.lambda2, aggregator.lambda3])
            weights = F.softmax(raw_lambdas, dim=0)

            student_feat = model.get_intermediate_features()  # (B, 128)

            if use_a and aligners[0] is not None and eA is not None and eA.shape[-1] == aligners[0].linear.in_features:
                tA = aligners[0](eA)
                kd_A_val = F.mse_loss(student_feat, tA)
                loss = loss + weights[0] * kd_A_val

            if use_b and aligners[1] is not None and eB is not None and eB.shape[-1] == aligners[1].linear.in_features:
                tB = aligners[1](eB)
                kd_B_val = F.mse_loss(student_feat, tB)
                loss = loss + weights[1] * kd_B_val

            if use_c and aligners[2] is not None and eC is not None and eC.shape[-1] == aligners[2].linear.in_features:
                tC = aligners[2](eC)
                kd_C_val = F.mse_loss(student_feat, tC)
                loss = loss + weights[2] * kd_C_val

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce.item()
        total_kd_A += kd_A_val.item()
        total_kd_B += kd_B_val.item()
        total_kd_C += kd_C_val.item()
        count += 1

    return {
        "total_loss": total_loss / max(count, 1),
        "ce_loss": total_ce / max(count, 1),
        "kd_A": total_kd_A / max(count, 1),
        "kd_B": total_kd_B / max(count, 1),
        "kd_C": total_kd_C / max(count, 1),
    }


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MT-LDI-MDS multi-teacher distillation trainer.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default 50).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default 32).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default 1e-4).")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu (default cuda).")
    parser.add_argument("--mode", type=str, default="learnable", choices=["fixed", "learnable"],
                        help="Aggregator mode (default learnable).")
    parser.add_argument("--kd-teachers", type=str, default="ABC",
                        help="Which teachers to use for KD: A, B, C, or any combination (default ABC).")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help="Path to save/load the best model checkpoint.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device = torch.device("cpu")

    print(f"[INFO] Using device: {device}")

    # ------------------------------------------------------------------
    # 1) Load data and split 70/15/15
    # ------------------------------------------------------------------
    print("[INFO] Loading VeReMi dataset ...")
    X, y = load_dataset()
    print(f"[INFO] Total samples: {len(y)} | Features: {X.shape[1]} | Classes: {NUM_CLASSES}")

    # 70 / 15 / 15 stratified split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )
    print(f"[INFO] Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # ------------------------------------------------------------------
    # 2) Load teacher embeddings (memory-mapped)
    # ------------------------------------------------------------------
    embs = load_teacher_embeddings(len(y_train))
    emb_A = embs.get("A")
    emb_B = embs.get("B")
    emb_C = embs.get("C")
    has_kd = any(e is not None for e in [emb_A, emb_B, emb_C])
    if not has_kd:
        print("[WARN] No teacher embeddings available — training with CE loss only.")

    # Determine teacher hidden dim from first available embedding array
    teacher_dim = 4096  # default Qwen3-8B hidden size
    for e in [emb_A, emb_B, emb_C]:
        if e is not None:
            teacher_dim = e.shape[1]
            break

    # ------------------------------------------------------------------
    # 3) Build DataLoaders
    # ------------------------------------------------------------------
    train_ds = TeacherEmbeddingDataset(X_train, y_train, emb_A, emb_B, emb_C)
    val_ds = TeacherEmbeddingDataset(X_val, y_val, None, None, None)
    test_ds = TeacherEmbeddingDataset(X_test, y_test, None, None, None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    # ------------------------------------------------------------------
    # 4) Build models
    # ------------------------------------------------------------------
    sys.path.insert(0, PROJECT_ROOT)
    from student.bilstm_student import BiLSTMStudent
    from aggregator.weighted_aggregator import WeightedAggregator

    input_dim = X_train.shape[1]
    student = BiLSTMStudent(input_dim=input_dim, num_classes=NUM_CLASSES).to(device)
    aggregator = WeightedAggregator(hidden_dim=teacher_dim, mode=args.mode, student_hidden_dim=256).to(device)

    # Teacher alignment heads (raw teacher_dim → 128 student intermediate dim)
    align_A = TeacherAligner(teacher_dim, 128).to(device) if emb_A is not None else None
    align_B = TeacherAligner(teacher_dim, 128).to(device) if emb_B is not None else None
    align_C = TeacherAligner(teacher_dim, 128).to(device) if emb_C is not None else None
    aligners = [align_A, align_B, align_C]

    # ------------------------------------------------------------------
    # 5) Optimizer & scheduler
    # ------------------------------------------------------------------
    params = list(student.parameters()) + list(aggregator.parameters())
    for al in aligners:
        if al is not None:
            params += list(al.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion_ce = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # 6) Training loop
    # ------------------------------------------------------------------
    best_val_acc = 0.0
    log_rows = []
    log_header = [
        "epoch", "total_loss", "ce_loss", "kd_loss_A", "kd_loss_B", "kd_loss_C",
        "val_accuracy", "val_f1_macro",
    ]

    os.makedirs(SCRIPT_DIR, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_metrics = train_epoch(student, aggregator, aligners, train_loader, optimizer, criterion_ce, device, has_kd, kd_teachers=args.kd_teachers)
        val_loss, val_ce, val_acc, val_f1 = evaluate(student, aggregator, aligners, val_loader, criterion_ce, device, kd_teachers=args.kd_teachers)

        scheduler.step()

        print(
            f"Train — total: {train_metrics['total_loss']:.4f} | "
            f"CE: {train_metrics['ce_loss']:.4f} | "
            f"KD_A: {train_metrics['kd_A']:.4f} | "
            f"KD_B: {train_metrics['kd_B']:.4f} | "
            f"KD_C: {train_metrics['kd_C']:.4f}"
        )
        print(f"Val   — loss: {val_loss:.4f} | CE: {val_ce:.4f} | Acc: {val_acc:.4f} | F1(macro): {val_f1:.4f}")

        # Print λ weights every 10 epochs
        if epoch % 10 == 0:
            aggregator.print_weights()

        # Checkpoint best model (by validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "student_state_dict": student.state_dict(),
                "aggregator_state_dict": aggregator.state_dict(),
                "aligners_state_dict": [al.state_dict() if al is not None else None for al in aligners],
                "val_accuracy": val_acc,
                "val_f1_macro": val_f1,
                "student_input_dim": input_dim,
                "teacher_dim": teacher_dim,
                "aggregator_mode": args.mode,
            }, args.checkpoint)
            print(f"[INFO] Saved new best model (val_acc={val_acc:.4f}) → {args.checkpoint}")

        log_rows.append({
            "epoch": epoch,
            "total_loss": train_metrics["total_loss"],
            "ce_loss": train_metrics["ce_loss"],
            "kd_loss_A": train_metrics["kd_A"],
            "kd_loss_B": train_metrics["kd_B"],
            "kd_loss_C": train_metrics["kd_C"],
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1,
        })

    # ------------------------------------------------------------------
    # 7) Save training log
    # ------------------------------------------------------------------
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_header)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[INFO] Training log saved → {LOG_PATH}")

    # ------------------------------------------------------------------
    # 8) Final test evaluation with best checkpoint
    # ------------------------------------------------------------------
    print("\n[INFO] Reloading best checkpoint for final test evaluation ...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["student_state_dict"])
    aggregator.load_state_dict(ckpt["aggregator_state_dict"])
    for al, sd in zip(aligners, ckpt["aligners_state_dict"]):
        if al is not None and sd is not None:
            al.load_state_dict(sd)

    test_loss, test_ce, test_acc, test_f1 = evaluate(student, aggregator, aligners, test_loader, criterion_ce, device, kd_teachers=args.kd_teachers)
    print(f"\n=== Final Test Results ===")
    print(f"Loss: {test_loss:.4f} | CE: {test_ce:.4f} | Accuracy: {test_acc:.4f} | F1(macro): {test_f1:.4f}")
    print(f"Best validation accuracy during training: {best_val_acc:.4f}")
    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()