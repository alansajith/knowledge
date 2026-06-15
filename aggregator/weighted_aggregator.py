"""
aggregator/weighted_aggregator.py
MT-LDI-MDS — Learnable Weighted Aggregator for Multi-Teacher Distillation

Fuses embeddings from three specialized teacher LLMs into a single
student-compatible representation. Supports:
  * fixed:   simple mean of (eA, eB, eC)
  * learnable: softmax-gated weighted sum with learnable λ parameters

After fusion the vector is passed through a linear adapter that projects
from teacher hidden dimension down to student_hidden_dim (256).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedAggregator(nn.Module):
    """
    Multi-teacher embedding aggregator.

    Args:
        hidden_dim: Dimensionality of each teacher embedding (e.g., 4096 for Qwen3-8B).
        mode: "fixed" or "learnable" weighting strategy.
        student_hidden_dim: Target dimension for the student network (default 256).
    """

    def __init__(self, hidden_dim: int, mode: str = "fixed", student_hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mode = mode.lower().strip()
        self.student_hidden_dim = student_hidden_dim

        if self.mode not in {"fixed", "learnable"}:
            raise ValueError(f"mode must be 'fixed' or 'learnable', got {mode}")

        # Learnable or fixed λ parameters (initialised uniformly at 1/3)
        if self.mode == "learnable":
            self.lambda1 = nn.Parameter(torch.tensor(1.0 / 3.0))
            self.lambda2 = nn.Parameter(torch.tensor(1.0 / 3.0))
            self.lambda3 = nn.Parameter(torch.tensor(1.0 / 3.0))
        else:
            # Fixed weights are stored as non-trainable buffers so get_weights() works uniformly
            self.register_buffer("lambda1", torch.tensor(1.0 / 3.0))
            self.register_buffer("lambda2", torch.tensor(1.0 / 3.0))
            self.register_buffer("lambda3", torch.tensor(1.0 / 3.0))

        # The spec mentions LinearAdapter(3*hidden_dim → student_hidden_dim).
        # Since both fixed and learnable modes produce a *weighted sum* of shape
        # (batch, hidden_dim), the adapter input is hidden_dim. If a future
        # variant requires concatenation first, change input_dim to 3*hidden_dim.
        self.adapter = nn.Linear(hidden_dim, student_hidden_dim)

    def forward(self, eA: torch.Tensor, eB: torch.Tensor, eC: torch.Tensor) -> torch.Tensor:
        """
        Fuse three teacher embeddings.

        Args:
            eA, eB, eC: Tensors of shape (batch_size, hidden_dim).

        Returns:
            Tensor of shape (batch_size, student_hidden_dim).
        """
        if eA.dim() != 2 or eB.dim() != 2 or eC.dim() != 2:
            raise ValueError("Expected 2-D inputs (batch, hidden_dim)")
        if eA.shape[1] != self.hidden_dim or eB.shape[1] != self.hidden_dim or eC.shape[1] != self.hidden_dim:
            raise ValueError(f"Input hidden dim mismatch: expected {self.hidden_dim}")

        if self.mode == "fixed":
            # Fixed mean aggregation
            fused = (eA + eB + eC) / 3.0
        else:
            # Learnable softmax-weighted aggregation
            weights = F.softmax(torch.stack([self.lambda1, self.lambda2, self.lambda3]), dim=0)
            fused = weights[0] * eA + weights[1] * eB + weights[2] * eC

        out = self.adapter(fused)
        return out

    def get_weights(self) -> dict:
        """
        Return current λ weights as a dict with float values.

        Returns:
            {"teacher_A": float, "teacher_B": float, "teacher_C": float}
        """
        with torch.no_grad():
            w = F.softmax(torch.stack([self.lambda1, self.lambda2, self.lambda3]), dim=0)
        return {
            "teacher_A": float(w[0].item()),
            "teacher_B": float(w[1].item()),
            "teacher_C": float(w[2].item()),
        }

    def print_weights(self):
        """Pretty-print current λ weights. Call from the training loop each epoch."""
        w = self.get_weights()
        print(
            f"[WeightedAggregator] λ weights — "
            f"A={w['teacher_A']:.4f}, B={w['teacher_B']:.4f}, C={w['teacher_C']:.4f}"
        )
