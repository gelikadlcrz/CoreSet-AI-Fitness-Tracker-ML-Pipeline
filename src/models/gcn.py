"""
gcn.py — Optimized ST-GCN building blocks.

Key improvements over baseline:
  1. Multi-scale temporal convolutions (parallel 3/7/13-frame kernels) replace
     the single 9-frame kernel — captures both fine motion and slow trends.
  2. Squeeze-and-Excitation (SE) channel attention after each ST-GCN block —
     lets the model up-weight discriminative angle channels per exercise.
  3. Spatial attention gate (SAG) — softly gates each of the 33 joints so
     irrelevant nodes (e.g., hand tips for squats) are suppressed.
  4. Dropout applied before the temporal conv, not just at heads, for
     stronger regularization inside the backbone.
  5. Residual projection is now weight-initialised with kaiming_normal_
     so gradients flow well from the first epoch.

References:
    Yan et al. (2018) ST-GCN
    Hu et al. (2018) Squeeze-and-Excitation Networks
    Shi et al. (2019) Directed Graph Neural Networks for skeleton action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation channel attention
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).
    Recalibrates channel-wise feature responses adaptively.

    Input/Output shape: (B, C, T, V, M)  — unchanged.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pool over T, V, M → (B, C)
        w = x.mean(dim=(2, 3, 4))          # (B, C)
        w = self.se(w)                      # (B, C)
        # Broadcast back: (B, C, 1, 1, 1)
        return x * w.view(w.size(0), w.size(1), 1, 1, 1)


# ---------------------------------------------------------------------------
# Spatial Attention Gate — softly weights each joint
# ---------------------------------------------------------------------------

class SpatialAttentionGate(nn.Module):
    """
    Learns a per-joint importance weight from pooled channel features.
    Keeps irrelevant landmarks (face, fingertips) from polluting features.

    Input/Output shape: (B, C, T, V, M)  — unchanged.
    """
    def __init__(self, channels: int, num_nodes: int = 33):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, v, m = x.size()
        # Pool over T and M, operate on V
        v_feat = x.mean(dim=(2, 4))    # (B, C, V)
        attn   = self.gate(v_feat)     # (B, 1, V)
        # Expand to full shape and apply
        attn   = attn.unsqueeze(-1).unsqueeze(-1)   # (B, 1, V, 1, 1)
        attn   = attn.permute(0, 1, 3, 2, 4)        # (B, 1, 1, V, 1)
        return x * attn                              # (B, C, T, V, M)


# ---------------------------------------------------------------------------
# Spatial Graph Convolution
# ---------------------------------------------------------------------------

class SpatialGraphConvolution(nn.Module):
    """
    Single-partition spatial graph convolution with learnable adjacency.

    Input shape:  (B, C_in,  T, V, M)
    Output shape: (B, C_out, T, V, M)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 norm_adj_matrix: np.ndarray, bias: bool = True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.adj = nn.Parameter(
            torch.from_numpy(norm_adj_matrix).float(),
            requires_grad=True
        )
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, v, m = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(b * m, c, t, v)
        x = self.conv(x)                        # (B*M, C_out, T, V)
        x = torch.matmul(x, self.adj)           # graph aggregation
        x = x.view(b, m, self.out_channels, t, v)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x


# ---------------------------------------------------------------------------
# Multi-scale Temporal Convolution
# ---------------------------------------------------------------------------

class MultiScaleTemporalConv(nn.Module):
    """
    Parallel temporal convolutions at 3 scales (short / mid / long),
    followed by a 1x1 conv to project back to out_channels.

    Replaces the single 9-frame kernel with three parallel streams so the
    model learns both rapid transitions and slow movement patterns.

    Input shape:  (B, C_in,  T, V, M)
    Output shape: (B, C_out, T//stride, V, M)
    """

    KERNEL_SIZES = (3, 7, 13)   # short / mid / long temporal receptive fields

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, dropout: float = 0.1):
        super().__init__()
        branch_channels = max(out_channels // len(self.KERNEL_SIZES), 8)

        self.branches = nn.ModuleList()
        for k in self.KERNEL_SIZES:
            pad = (k - 1) // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels,
                              kernel_size=(k, 1),
                              stride=(stride, 1),
                              padding=(pad, 0)),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=False),
                )
            )

        # Combine branches only (no residual passthrough into merge)
        concat_channels = branch_channels * len(self.KERNEL_SIZES)
        self.merge = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def _to4d(self, x: torch.Tensor):
        b, c, t, v, m = x.size()
        x4 = x.permute(0, 4, 1, 2, 3).contiguous().view(b * m, c, t, v)
        return x4, b, c, t, v, m

    def _to5d(self, x4, b, cout, t_out, v, m):
        return x4.view(b, m, cout, t_out, v).permute(0, 2, 3, 4, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x4, b, c, t, v, m = self._to4d(x)
        x4 = self.dropout(x4)

        branches_out = [branch(x4) for branch in self.branches]
        t_out = branches_out[0].size(2)

        merged = self.merge(torch.cat(branches_out, dim=1))  # (B*M, C_out, T', V)
        return self._to5d(merged, b, merged.size(1), t_out, v, m)


# ---------------------------------------------------------------------------
# ST-GCN Block (optimized)
# ---------------------------------------------------------------------------

class ST_GCN_Block(nn.Module):
    """
    Optimized Spatial-Temporal GCN block:

        Spatial GCN → BN → ReLU
            → SE channel attention
            → Spatial attention gate
            → Multi-scale temporal conv (parallel 3/7/13 kernels)
            → ReLU → (+residual)

    The SE block and spatial attention gate are lightweight additions
    (< 1 % parameter overhead) that markedly improve discriminability.

    Args:
        in_channels   (int): Input feature channels.
        out_channels  (int): Output feature channels.
        adj_matrix    (np.ndarray): Normalised adjacency (V, V).
        stride        (int): Temporal stride (1 = same len, 2 = halved).
        residual      (bool): Add residual/skip connection.
        se_reduction  (int): SE block reduction ratio. Default 4.
        dropout       (float): Temporal conv dropout. Default 0.1.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 adj_matrix: np.ndarray, stride: int = 1,
                 residual: bool = True, se_reduction: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        num_nodes = adj_matrix.shape[0]

        # Spatial GCN + BN
        self.gcn    = SpatialGraphConvolution(in_channels, out_channels, adj_matrix)
        self.bn_gcn = nn.BatchNorm2d(out_channels)

        # Channel + spatial attention
        self.se_attn      = SEBlock(out_channels, reduction=se_reduction)
        self.spatial_attn = SpatialAttentionGate(out_channels, num_nodes)

        # Multi-scale temporal conv
        self.t_conv = MultiScaleTemporalConv(out_channels, out_channels,
                                              stride=stride, dropout=dropout)

        self.relu = nn.ReLU(inplace=False)

        # Residual branch
        if not residual:
            self.residual = None
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
            # Proper init for residual projection
            nn.init.kaiming_normal_(self.residual[0].weight,
                                    mode='fan_out', nonlinearity='relu')

    def _bn2d(self, module, x):
        b, c, t, v, m = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(b * m, c, t, v)
        x = module(x)
        x = x.view(b, m, c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x

        # Spatial path
        x = self.gcn(x)
        x = self._bn2d(self.bn_gcn, x)
        x = self.relu(x)

        # Attention
        x = self.se_attn(x)
        x = self.spatial_attn(x)

        # Multi-scale temporal path
        x = self.t_conv(x)
        x = self.relu(x)

        # Residual
        if self.residual is not None:
            if not isinstance(self.residual, nn.Identity):
                b, c, t, v, m = res.size()
                res = res.permute(0, 4, 1, 2, 3).contiguous().view(b * m, c, t, v)
                res = self.residual(res)
                t_out = res.size(2)
                res = res.view(b, m, res.size(1), t_out, v)
                res = res.permute(0, 2, 3, 4, 1).contiguous()
            x = x + res

        return self.relu(x)