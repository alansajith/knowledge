"""
data/split_dataset.py
MT-LDI-MDS — Data Preprocessing Pipeline

This script:
1. Loads the VeReMi extension dataset.
2. Maps string attack_type labels to numeric labels 0–8.
3. Splits into three teacher subsets (A, B, C).
4. Applies Z-score outlier removal.
5. Trains a small projection head with Supervised Contrastive Learning (SCL)
   to cluster attack embeddings within each subset.
6. Finds top-5 nearest same-class neighbors (in the SCL embedding space)
   and records mean distance as nn_variability.
7. Saves splits and prints class distributions.
"""

import argparse
import os
import sys

# Allow importing utils regardless of where the script is launched from
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import zscore
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.paths import get_project_paths

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
paths = get_project_paths(__file__)
DATA_DIR = paths["DATA_DIR"]
SOURCE_CSV = os.path.join(DATA_DIR, "veremi.csv")

# Numeric label mapping (0=benign handled separately)
ATTACK_MAP = {
    "DoS": 1,
    "DoSDisruptive": 1,
    "DoSRandom": 1,
    "DoSRandomSybil": 1,
    "DoSDisruptiveSybil": 1,
    "Disruptive": 1,
    "DelayedMessages": 1,  # interpreted as a DoS-like disruption variant
    "GridSybil": 2,
    "ConstPosOffset": 3,
    "ConstPos": 3,
    "RandomPosOffset": 4,
    "RandomPos": 4,
    "EventualStop": 5,
    "ConstSpeedOffset": 6,
    "ConstSpeed": 6,
    "RandomSpeedOffset": 7,
    "RandomSpeed": 7,
    "DataReplaySybil": 8,
    "DataReplay": 8,
}

TEACHER_SPLITS = {
    "A": [0, 1, 2],
    "B": [0, 3, 4, 5],
    "C": [0, 6, 7, 8],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Custom Supervised Contrastive Loss (SupConLoss)
# ---------------------------------------------------------------------------
class SupConLoss(nn.Module):
    """
    Lightweight Supervised Contrastive Loss.

    For each anchor, pull positives (same class) closer and push negatives
    (different class) away in the normalized embedding space.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feat_dim) — already projected embeddings.
            labels:   (batch_size,) — integer class labels.

        Returns:
            Scalar loss.
        """
        device = features.device
        batch_size = features.shape[0]

        # L2-normalize so that dot product becomes cosine similarity
        features = F.normalize(features, dim=1)

        # Compute pairwise cosine similarities scaled by temperature
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Mask out self-similarity (diagonal)
        mask_self = torch.eye(batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask_self, -9e15)

        # Build positive mask: same label, excluding self
        labels_expanded = labels.unsqueeze(0)  # (1, B)
        pos_mask = (labels_expanded == labels_expanded.T).float()          # (B, B)
        pos_mask = pos_mask.masked_fill(mask_self, 0.0)

        # Numerical stability: subtract max per row before exp
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_stable = sim_matrix - sim_max.detach()

        exp_sim = torch.exp(sim_stable)
        log_prob = sim_stable - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # Average log-prob over positives for each anchor
        pos_count = pos_mask.sum(dim=1) + 1e-12
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count

        loss = -mean_log_prob_pos.mean()
        return loss


# ---------------------------------------------------------------------------
# SCL Projection Head
# ---------------------------------------------------------------------------
class SCLProjectionHead(nn.Module):
    """Small MLP that projects raw vehicular features into an SCL embedding space."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_and_label_dataset(csv_path: str) -> pd.DataFrame:
    """Load VeReMi CSV, drop the spurious index column, and add numeric_label."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Please download the VeReMi Extension dataset from\n"
            "https://github.com/josephkamel/VeReMi-Dataset\n"
            "and place it at the expected path."
        )

    df = pd.read_csv(csv_path, low_memory=False)

    # Drop spurious index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Map string attack_type to numeric labels
    # Benign rows have attack=0 and NaN attack_type → label 0
    df["numeric_label"] = df["attack_type"].map(ATTACK_MAP).fillna(0).astype(int)

    # Sanity check: all attack=1 rows should map to 1-8
    attack_mask = df["attack"] == 1
    unmapped = df.loc[attack_mask, "numeric_label"] == 0
    if unmapped.any():
        bad_types = df.loc[attack_mask & unmapped, "attack_type"].unique()
        raise ValueError(f"Unmapped attack types found: {bad_types}")

    return df


def get_numerical_cols(df: pd.DataFrame) -> list:
    """Return columns that are numeric and not metadata/labels."""
    exclude = {"attack", "attack_type", "numeric_label", "nn_variability"}
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


def zscore_outlier_removal(df: pd.DataFrame, cols: list, threshold: float = 3.0) -> pd.DataFrame:
    """Remove rows where ANY numerical feature deviates |z| > threshold."""
    # Compute z-scores on the selected columns (ignoring NaNs)
    z_vals = np.abs(zscore(df[cols], nan_policy="omit"))
    mask = (z_vals < threshold).all(axis=1)
    before = len(df)
    df_clean = df[mask].reset_index(drop=True)
    after = len(df_clean)
    if before != after:
        print(f"  Z-score outlier removal: dropped {before - after} rows (threshold={threshold})")
    return df_clean


def train_scl(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: torch.device = DEVICE,
) -> SCLProjectionHead:
    """Train a projection head with SupConLoss on the given feature matrix."""
    model = SCLProjectionHead(input_dim=input_dim, hidden_dim=128, output_dim=64).to(device)
    criterion = SupConLoss(temperature=0.07)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for batch_x, batch_y in tqdm(loader, desc=f"  SCL epoch {epoch+1}/{epochs}", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            proj = model(batch_x)
            loss = criterion(proj, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        print(f"  SCL epoch {epoch+1}/{epochs} — avg loss: {avg_loss:.4f}")

    return model


def compute_nn_variability(X: np.ndarray, labels: np.ndarray, k: int = 5) -> np.ndarray:
    """
    For each class, compute k nearest neighbors in the (already projected) feature space.
    Return the mean Euclidean distance to the k neighbors for every sample.
    """
    n_samples = X.shape[0]
    variability = np.zeros(n_samples, dtype=np.float32)

    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        idxs = np.where(mask)[0]
        X_cls = X[mask]

        nn = NearestNeighbors(n_neighbors=min(k + 1, len(X_cls)), metric="euclidean", algorithm="ball_tree")
        nn.fit(X_cls)
        distances, _ = nn.kneighbors(X_cls)
        # Exclude self (first neighbor distance == 0)
        mean_dist = distances[:, 1:].mean(axis=1) if distances.shape[1] > 1 else 0.0
        variability[idxs] = mean_dist

    return variability


def split_and_preprocess(df: pd.DataFrame, teacher: str, max_per_class: int = 100_000):
    """
    Filter rows belonging to the teacher's labels, apply Z-score removal,
    optional per-class subsampling, train SCL, compute nn_variability.
    """
    labels_needed = TEACHER_SPLITS[teacher]
    df_sub = df[df["numeric_label"].isin(labels_needed)].copy()
    print(f"\nTeacher {teacher}: {len(df_sub)} raw samples (labels {labels_needed})")

    # ------------------------------------------------------------------
    # 1) Z-score outlier removal on raw numerical features
    # ------------------------------------------------------------------
    num_cols = get_numerical_cols(df_sub)
    df_sub = zscore_outlier_removal(df_sub, num_cols, threshold=3.0)
    print(f"  After Z-score cleaning: {len(df_sub)} samples")

    # ------------------------------------------------------------------
    # 2) Optional per-class subsampling
    # ------------------------------------------------------------------
    sampled = []
    for lbl in labels_needed:
        df_cls = df_sub[df_sub["numeric_label"] == lbl]
        if len(df_cls) > max_per_class:
            df_cls = df_cls.sample(n=max_per_class, random_state=42)
            print(f"    Class {lbl}: subsampled to {max_per_class} (from {len(df_sub[df_sub['numeric_label']==lbl])})")
        sampled.append(df_cls)
    df_sub = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  After subsampling: {len(df_sub)} samples")

    # ------------------------------------------------------------------
    # 3) Extract numerical features for SCL
    # ------------------------------------------------------------------
    X_raw = df_sub[num_cols].to_numpy(dtype=np.float32)
    y_raw = df_sub["numeric_label"].to_numpy(dtype=np.int64)

    # Normalize raw features to zero mean / unit variance (helps SCL converge)
    feat_mean = X_raw.mean(axis=0, keepdims=True)
    feat_std = X_raw.std(axis=0, keepdims=True) + 1e-12
    X_norm = (X_raw - feat_mean) / feat_std

    # ------------------------------------------------------------------
    # 4) Train SCL projection head
    # ------------------------------------------------------------------
    print(f"  Training SCL projection head on {X_norm.shape[0]} samples ({X_norm.shape[1]} dims) ...")
    proj_model = train_scl(X_norm, y_raw, input_dim=X_norm.shape[1], epochs=10, batch_size=512, lr=1e-3)

    # ------------------------------------------------------------------
    # 5) Compute SCL embeddings and nearest-neighbor variability
    # ------------------------------------------------------------------
    proj_model.eval()
    with torch.no_grad():
        # Process in mini-batches to avoid OOM on large subsets
        batches = []
        batch_sz = 4096
        for i in range(0, len(X_norm), batch_sz):
            xb = torch.from_numpy(X_norm[i : i + batch_sz]).to(DEVICE)
            eb = proj_model(xb).cpu().numpy()
            batches.append(eb)
    X_proj = np.concatenate(batches, axis=0)

    print("  Computing top-5 nearest same-class neighbors ...")
    nn_var = compute_nn_variability(X_proj, y_raw, k=5)
    df_sub["nn_variability"] = nn_var

    # ------------------------------------------------------------------
    # 6) Class distribution report
    # ------------------------------------------------------------------
    print("  Class distribution:")
    print(df_sub["numeric_label"].value_counts().sort_index().to_string())

    return df_sub


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare VeReMi data splits for MT-LDI-MDS teachers."
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=100_000,
        help="Maximum samples to keep per class after Z-score cleaning (default: 100000). "
             "Set to 0 to disable subsampling.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 0) Handle missing dataset — copy from repo root if needed
    # ------------------------------------------------------------------
    if not os.path.exists(SOURCE_CSV):
        fallback = os.path.join(paths["PROJECT_ROOT"], "Veremi_final_dataset.csv")
        if os.path.exists(fallback):
            print(f"Copying {fallback} → {SOURCE_CSV}")
            os.makedirs(paths["DATA_DIR"], exist_ok=True)
            import shutil
            shutil.copy2(fallback, SOURCE_CSV)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {SOURCE_CSV}\n"
                "Please download the VeReMi Extension dataset from\n"
                "https://github.com/josephkamel/VeReMi-Dataset\n"
                "and place it at ./data/veremi.csv or ./Veremi_final_dataset.csv"
            )

    # ------------------------------------------------------------------
    # 1) Load + label
    # ------------------------------------------------------------------
    print("Loading and labeling VeReMi dataset ...")
    df = load_and_label_dataset(SOURCE_CSV)
    print(f"Total samples: {len(df)} | Features: {len(get_numerical_cols(df))}")
    print("Global label distribution:")
    print(df["numeric_label"].value_counts().sort_index().to_string())

    # ------------------------------------------------------------------
    # 2) Process each teacher subset
    # ------------------------------------------------------------------
    max_per_class = args.max_per_class
    for teacher in ["A", "B", "C"]:
        df_teacher = split_and_preprocess(df, teacher, max_per_class=max_per_class)
        out_path = os.path.join(DATA_DIR, f"split_{teacher}.csv")
        df_teacher.to_csv(out_path, index=False)
        print(f"Saved Teacher {teacher} split → {out_path} ({len(df_teacher)} rows)")

    print("\n✓ All splits generated successfully.")


if __name__ == "__main__":
    main()