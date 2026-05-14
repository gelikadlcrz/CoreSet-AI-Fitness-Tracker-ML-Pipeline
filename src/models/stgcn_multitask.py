"""
stgcn_multitask.py — Optimized Multi-task ST-GCN (v3).

Rep-counting overhaul
---------------------
The original density head failed because of a representation mismatch:
  - GT density sums to rep_count (e.g. 8.0 over 150 frames → peak ≈ 0.05)
  - Model outputs sigmoid ∈ [0, 1]
  - MSE drives predictions toward zero (predicting 0 everywhere is near-optimal
    for MSE when peaks are tiny fractions)
  - Peak detection used threshold=0.5 → found zero peaks → OBO=9%

Fix: train on NORMALIZED density maps (GT divided by its max, so peaks=1.0).
  - Model learns peak SHAPE and LOCATION, not absolute magnitude
  - Peak detection threshold is meaningful (0.3 on a 0–1 scale)
  - Rep count = number of detected peaks (not density integral)
  - The coreset_dataset.py __getitem__ returns the raw rep count in the npz
    so the evaluator can recover it without integral-of-density tricks

Density head: lightweight 1-D dilated TCN (5 layers, dilations 1/2/4/8/1)
  - Replaces the Transformer which was unstable on 700 samples
  - Receptive field ≈ 65 frames — covers ~2 full reps
  - weight_norm Conv1d for stable training
  - Output: sigmoid ∈ [0,1] = normalized peak probability

Classification head: 2-layer MLP with LayerNorm+GELU.

This revision lowers head dropout slightly so the small 700-sample dataset keeps
stronger class-discriminative features while still regularizing overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.graph_utils import build_blazepose_33_adjacency_matrix
from src.models.gcn import ST_GCN_Block


# ---------------------------------------------------------------------------
# Lightweight dilated TCN density head
# ---------------------------------------------------------------------------

class DilatedTCNDensityHead(nn.Module):
    """
    5-layer non-causal dilated TCN for density-map regression.

    Effective receptive field = 1 + 2*(1+2+4+8+1)*(kernel_size-1) = 65 frames
    with kernel_size=3. Covers roughly 2 complete reps at 30 fps.

    Input:  (B, in_channels, T')
    Output: (B, max_frames) in [0, 1]
    """
    def __init__(self, in_channels: int = 256, d: int = 64,
                 kernel_size: int = 3, max_frames: int = 150,
                 dropout: float = 0.1):
        super().__init__()
        self.max_frames = max_frames

        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, d, kernel_size=1, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(),
        )

        dilations = [1, 2, 4, 8, 1]
        layers = []
        for dil in dilations:
            pad = dil * (kernel_size - 1) // 2
            layers += [
                nn.utils.weight_norm(
                    nn.Conv1d(d, d, kernel_size=kernel_size,
                              padding=pad, dilation=dil, bias=False)
                ),
                nn.BatchNorm1d(d),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
        self.tcn = nn.Sequential(*layers)
        self.out = nn.Conv1d(d, 1, kernel_size=1)
        self.drop_in = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_in(x)
        x = self.proj(x)
        x = self.tcn(x)
        x = self.out(x)
        x = F.interpolate(x, size=self.max_frames,
                          mode='linear', align_corners=False)
        return torch.sigmoid(x.squeeze(1))


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int,
                 mid_features: int = 128, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, mid_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mid_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Multi-scale input smoothing
# ---------------------------------------------------------------------------

class MultiScaleInputSmoothing(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.short = nn.Conv2d(channels, channels,
                               kernel_size=(5, 1), padding=(2, 0), bias=False)
        self.long  = nn.Conv2d(channels, channels,
                               kernel_size=(13, 1), padding=(6, 0), bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, v, m = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(b * m, c, t, v)
        x = self.bn(self.short(x) + self.long(x))
        x = x.view(b, m, c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CoreSetSTGCN_MultiTask(nn.Module):
    """
    Multi-task ST-GCN v3 — optimized backbone + dilated TCN density head.

    IMPORTANT — density GT normalization contract:
        The density head is trained on GT maps normalized to [0,1] by dividing
        by their per-sample maximum. Do this in the training loop:
            gt_norm = normalize_density_gt(density_gts)   # see train_stgcn.py
        The model predicts normalized peak shape; rep count is recovered by
        counting peaks above threshold in the predicted map.
    """

    CHANNELS = [32, 64, 128, 256]

    def __init__(self, in_channels: int = 14, num_classes: int = 4,
                 max_frames: int = 150, node_count: int = 33):
        super().__init__()
        self.num_classes = num_classes
        self.max_frames  = max_frames

        adj = build_blazepose_33_adjacency_matrix()

        self.input_smoothing = MultiScaleInputSmoothing(in_channels)

        c = self.CHANNELS
        self.st_gcn_1 = ST_GCN_Block(in_channels, c[0], adj, stride=1, residual=False)
        self.st_gcn_2 = ST_GCN_Block(c[0], c[1], adj, stride=2)
        self.st_gcn_3 = ST_GCN_Block(c[1], c[2], adj, stride=2)
        self.st_gcn_4 = ST_GCN_Block(c[2], c[3], adj, stride=2)

        final_ch = c[-1]

        self.head_classification = ClassificationHead(
            in_features=final_ch, num_classes=num_classes,
            mid_features=128, dropout=0.35
        )
        self.head_density = DilatedTCNDensityHead(
            in_channels=final_ch, d=64,
            kernel_size=3, max_frames=max_frames, dropout=0.08
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                # weight_norm wraps the weight; skip those to avoid AttributeError
                if hasattr(m, 'weight') and m.weight is not None:
                    try:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                                nonlinearity='relu')
                    except ValueError:
                        pass
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.input_smoothing(x)

        x = self.st_gcn_1(x)
        x = self.st_gcn_2(x)
        x = self.st_gcn_3(x)
        x = self.st_gcn_4(x)

        cls_feat    = x.mean(dim=(2, 3, 4))
        logits      = self.head_classification(cls_feat)

        density_feat = x.mean(dim=(3, 4))
        density_map  = self.head_density(density_feat)

        return logits, density_map