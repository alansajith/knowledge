"""
student/bilstm_student.py
MT-LDI-MDS — BiLSTM Student Architecture

Exact architecture from the base paper:
  1) Input embedding layer (input_dim → 128)
  2) Two Conv1D layers (filters=64, kernel=3, padding=same) + ReLU
  3) Permute to rearrange for LSTM input
  4) Three stacked BiLSTM layers (hidden=128, bidirectional=True, dropout=0.3)
  5) Multi-head self-attention (num_heads=4, embed_dim=256)
  6) Residual connection + LayerNorm
  7) Global Average Pooling
  8) FC layer (256 → 128) + BatchNorm + Dropout(0.3) + ReLU
  9) Output layer (128 → num_classes=9)

get_intermediate_features() returns the 128-dim vector (after BatchNorm) used
for feature-based multi-teacher knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMStudent(nn.Module):
    """
    BiLSTM-based lightweight student for vehicular misbehavior detection.

    Args:
        input_dim: Number of raw features per BSM sample (e.g., 18 for VeReMi).
        num_classes: Number of output classes (default 9 for VeReMi labels).
    """

    def __init__(self, input_dim: int, num_classes: int = 9):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self._intermediate = None

        # ------------------------------------------------------------------
        # 1) Input embedding layer (input_dim → 128)
        #    Applied independently to each timestep: (B, T, input_dim) → (B, T, 128)
        # ------------------------------------------------------------------
        self.embedding = nn.Linear(input_dim, 128)

        # ------------------------------------------------------------------
        # 2) Two Conv1D layers (filters=64, kernel=3, padding="same")
        #    PyTorch Conv1d expects (B, channels, length).
        #    We permute (B, T, 128) → (B, 128, T), apply conv, then permute back.
        # ------------------------------------------------------------------
        self.conv1 = nn.Conv1d(
            in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1
        )
        self.conv_relu = nn.ReLU(inplace=True)

        # ------------------------------------------------------------------
        # 3) Permute layer is implicit in forward (permute back for LSTM).
        # 4) Three stacked BiLSTM layers
        #    input_size = 64 (from conv output), hidden_size = 128 per direction
        #    bidirectional → output dim = 256
        # ------------------------------------------------------------------
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # ------------------------------------------------------------------
        # 5) Multi-head self-attention (embed_dim = 256, num_heads = 4)
        # 6) Residual + LayerNorm
        # ------------------------------------------------------------------
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=4, batch_first=True
        )
        self.ln = nn.LayerNorm(256)

        # ------------------------------------------------------------------
        # 7) Global Average Pooling over the sequence dimension
        # 8) FC (256 → 128) + BatchNorm1d + Dropout(0.3) + ReLU
        # ------------------------------------------------------------------
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)

        # ------------------------------------------------------------------
        # 9) Output layer (128 → num_classes)
        # ------------------------------------------------------------------
        self.fc_out = nn.Linear(128, num_classes)

    # ------------------------------------------------------------------
    # Feature extraction helper (shared by forward and get_intermediate_features)
    # ------------------------------------------------------------------
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the feature extraction pipeline up to the 128-dim hidden layer.

        Input shape:  (batch, seq_len, input_dim)
        Output shape: (batch, 128)
        """
        # 1) Embedding
        x = self.embedding(x)  # (B, T, 128)

        # 2) Conv1D — permute to (B, channels=128, length=T)
        x = x.permute(0, 2, 1)
        x = self.conv_relu(self.conv1(x))  # (B, 64, T)
        x = self.conv_relu(self.conv2(x))  # (B, 64, T)

        # 3) Permute back to (B, T, 64) for LSTM
        x = x.permute(0, 2, 1)

        # 4) BiLSTM
        x, _ = self.bilstm(x)  # (B, T, 256)

        # 5) Multi-head self-attention on the LSTM output
        attn_out, _ = self.attention(x, x, x)  # (B, T, 256)

        # 6) Residual connection + LayerNorm
        x = self.ln(attn_out + x)  # (B, T, 256)

        # 7) Global Average Pooling over sequence length T
        x = x.mean(dim=1)  # (B, 256)

        # 8) FC → 128 + BN + Dropout + ReLU
        x = self.fc1(x)       # (B, 128)
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(x, inplace=True)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass returning classification logits.

        Input shape:  (batch, seq_len, input_dim)
        Output shape: (batch, num_classes)
        """
        feat = self._extract_features(x)
        self._intermediate = feat  # cache for KD
        out = self.fc_out(feat)    # (B, num_classes)
        return out

    def get_intermediate_features(self) -> torch.Tensor:
        """
        Return the 128-dim pre-classification feature vector produced during
        the most recent forward() call (after BatchNorm, before fc_out).

        Raises:
            RuntimeError: if forward() has not been called yet.
        """
        if self._intermediate is None:
            raise RuntimeError(
                "get_intermediate_features() called before forward(). "
                "Run forward(x) first."
            )
        return self._intermediate
