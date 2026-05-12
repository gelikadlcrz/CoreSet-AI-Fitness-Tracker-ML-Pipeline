import torch
import torch.nn as nn


class Attention(nn.Module):
    """Simple additive (Bahdanau-style) attention over time steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, H)
        Returns:
            context : (B, H)  — attention-weighted summary
            weights : (B, T)  — normalised attention weights
        """
        scores = self.attn(x).squeeze(-1)          # (B, T)
        weights = torch.softmax(scores, dim=1)     # (B, T)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, H)
        return context, weights


class MultiTaskBiLSTM(nn.Module):
    """
    Bidirectional LSTM baseline for exercise classification and
    repetition counting.

    Architecture (per methodology):
    - Two stacked bidirectional LSTM layers, 128 hidden units per direction
    - Additive attention pooling over the temporal output
    - Classification head  → 4-class exercise taxonomy (cross-entropy)
    - Regression head      → rep-count scalar           (MSE)
    - Dropout = 0.5 before each final linear layer

    Input tensors are assumed to be pre-standardised (zero mean, unit
    variance) by the offline pre-processing pipeline; therefore no
    LayerNorm on the input is required.  All sequences in a batch share
    the same fixed temporal length after pre-processing, so no
    pack_padded_sequence machinery is needed.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            # Inter-layer dropout (only applied when num_layers > 1)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2          # bidirectional concatenation

        self.attention = Attention(lstm_out_dim)
        self.layer_norm = nn.LayerNorm(lstm_out_dim)

        # Dropout applied before each head's final linear layer
        # (rate = 0.5, matching methodology specification)
        self.dropout = nn.Dropout(dropout)

        # ── Classification head ──────────────────────────────────────
        # Predicts exercise class; supervised by categorical cross-entropy.
        self.cls_head = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        # ── Regression head ──────────────────────────────────────────
        # Predicts repetition count as a continuous scalar;
        # supervised by MSE loss (λ = 0.5).
        self.reg_head = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_size)  — pre-standardised joint-angle sequences

        Returns:
            logits  : (B, num_classes)   classification logits
            rep_out : (B, 1)             predicted repetition count
        """
        lstm_out, _ = self.lstm(x)              # (B, T, 2*hidden_size)

        features, _ = self.attention(lstm_out)  # (B, 2*hidden_size)
        features = self.layer_norm(features)
        features = self.dropout(features)

        logits = self.cls_head(features)        # (B, num_classes)
        rep_out = self.reg_head(features)       # (B, 1)

        return logits, rep_out