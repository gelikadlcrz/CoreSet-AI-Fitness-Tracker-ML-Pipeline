"""
stgcn_multitask.py — Multi-task ST-GCN for exercise classification and
repetition density-map prediction.

Architecture:
    Input smoothing (learned 1-D temporal conv at input stage)
        ↓
    ST-GCN backbone (3 blocks: in_channels→32→64→128, blocks 2/3 stride-2)
        ↓
    ┌─────────────────────────────────┐
    │  Head 1: Classification         │
    │  GlobalAvgPool → Dropout(0.5)   │
    │  → Linear(128, num_classes)     │
    └─────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────────────┐
    │  Head 2: Density map                                            │
    │  SpatialAvgPool (→ temporal series) → Dropout(0.3)              │
    │  → Conv1d(128,1,k=1) → interpolate to max_frames → Sigmoid      │
    └─────────────────────────────────────────────────────────────────┘

MPS compatibility note:
    AdaptiveAvgPool3d is not implemented on Apple MPS (as of PyTorch 2.x).
    All global and spatial pooling is therefore done via explicit .mean()
    calls over named dimensions, which are fully MPS-compatible.

References:
    Yan et al. (2018) ST-GCN
    Hu et al. (2022) TransRAC / RepCount density-map supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.graph_utils import build_blazepose_33_adjacency_matrix
from src.models.gcn import ST_GCN_Block


class CoreSetSTGCN_MultiTask(nn.Module):
    """
    Multi-task Spatial-Temporal GCN.

    Args:
        in_channels (int): Node feature channels (ANGLE_FEATURE_DIM = 14).
        num_classes (int): Number of exercise classes. Default 4.
        max_frames  (int): Fixed temporal window after padding/truncation.
        node_count  (int): Number of skeleton nodes (33 for BlazePose).
    """

    def __init__(self, in_channels: int = 14, num_classes: int = 4,
                 max_frames: int = 150, node_count: int = 33):
        super().__init__()

        self.num_classes = num_classes
        self.max_frames = max_frames

        # Symmetrically-normalised anatomical adjacency matrix
        adj = build_blazepose_33_adjacency_matrix()

        # ------------------------------------------------------------------ #
        #  Input smoothing                                                     #
        #  "Learned 1D temporal convolutional layer at the input stage"        #
        #  Kernel (9, 1): spans 9 frames × 1 node — purely temporal.          #
        # ------------------------------------------------------------------ #
        self.input_smoothing = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(9, 1), padding=(4, 0), bias=False
        )
        self.input_smoothing_bn = nn.BatchNorm2d(in_channels)

        # ------------------------------------------------------------------ #
        #  ST-GCN backbone                                                     #
        # ------------------------------------------------------------------ #
        self.st_gcn_1 = ST_GCN_Block(
            in_channels, 32, adj_matrix=adj, residual=False
        )
        self.st_gcn_2 = ST_GCN_Block(32,  64,  adj_matrix=adj, stride=2)
        self.st_gcn_3 = ST_GCN_Block(64,  128, adj_matrix=adj, stride=2)

        # ------------------------------------------------------------------ #
        #  Head 1: Classification                                              #
        #  Global average pool implemented as .mean() — MPS compatible.       #
        # ------------------------------------------------------------------ #
        self.head_classification = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

        # ------------------------------------------------------------------ #
        #  Head 2: Density map                                                 #
        #  Spatial pool over (V, M) via .mean() — MPS compatible.             #
        # ------------------------------------------------------------------ #
        self.head_density = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Conv1d(128, 1, kernel_size=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _apply_input_smoothing(self, x: torch.Tensor) -> torch.Tensor:
        """Learned temporal smoothing in (B*M, C, T, V) space."""
        b, c, t, v, m = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(b * m, c, t, v)
        x = self.input_smoothing_bn(self.input_smoothing(x))
        x = x.view(b, m, c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        return x

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, T, V, M)

        Returns:
            classification_logits: (B, num_classes)
            density_map:           (B, max_frames), values in [0, 1]
        """
        # --- Input smoothing ---
        x = self._apply_input_smoothing(x)     # (B, C,   T,    V, M)

        # --- ST-GCN backbone ---
        x = self.st_gcn_1(x)                   # (B, 32,  T,    V, M)
        x = self.st_gcn_2(x)                   # (B, 64,  T//2, V, M)
        x = self.st_gcn_3(x)                   # (B, 128, T//4, V, M)

        # x shape: (B, C=128, T', V=33, M=1)

        # ------------------------------------------------------------------ #
        #  Classification head                                                 #
        #  Global average pool over T', V, M → (B, 128)                      #
        #  Using .mean() instead of AdaptiveAvgPool3d for MPS compatibility.  #
        # ------------------------------------------------------------------ #
        # mean over T' (dim=2), then V (dim=2 after first mean), then M
        cls_feat = x.mean(dim=2).mean(dim=2).mean(dim=2)   # (B, 128)
        classification_logits = self.head_classification(cls_feat)

        # ------------------------------------------------------------------ #
        #  Density-map head                                                    #
        #  Pool over V (dim=3) and M (dim=4), preserve T' (dim=2).           #
        #  Using .mean() for MPS compatibility.                               #
        # ------------------------------------------------------------------ #
        density = x.mean(dim=4).mean(dim=3)    # (B, 128, T')

        density = self.head_density(density)    # (B, 1, T')

        # Upsample compressed temporal dim back to max_frames for MSE loss
        density = F.interpolate(
            density,
            size=self.max_frames,
            mode='linear',
            align_corners=False
        )                                       # (B, 1, max_frames)

        density_map = torch.sigmoid(density.squeeze(1))   # (B, max_frames)

        return classification_logits, density_map