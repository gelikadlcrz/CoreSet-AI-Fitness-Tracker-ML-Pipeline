"""
gcn.py — Spatial-Temporal Graph Convolutional Network building blocks.

Architecture follows Yan et al. (2018) "Spatial Temporal Graph Convolutional
Networks for Skeleton-Based Action Recognition" with the adaptive graph
extension from Shi et al. (2019) "Skeleton-Based Action Recognition with
Directed Graph Neural Networks."

Key differences from a naive ST-GCN:
  - Learnable edge weights (nn.Parameter) rather than a fixed buffer, allowing
    the model to refine connectivity beyond the hard-coded anatomical prior,
    as described in the methodology and Shi et al. (2019).
  - BatchNorm2d applied after BOTH the spatial and temporal convolutions
    inside every ST_GCN_Block, matching the original Yan et al. implementation.
  - Symmetric normalisation of the adjacency matrix (D^{-1/2} A D^{-1/2}).
  - All 5D tensor reshaping (b, c, t, v, m) is explicit and consistent.
"""

import torch
import torch.nn as nn
import numpy as np


class SpatialGraphConvolution(nn.Module):
    """
    Single-partition spatial graph convolution.

    Implements: Z = (X · W) · Â
    where Â is a learnable, symmetrically-normalised adjacency matrix
    that is initialised from the anatomical prior but updated during
    training (Shi et al., 2019).

    Input shape:  (B, C_in,  T, V, M)
    Output shape: (B, C_out, T, V, M)

    B = batch, C = channels, T = frames, V = vertices (joints), M = persons.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 norm_adj_matrix: np.ndarray, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- Learnable adjacency matrix (Shi et al., 2019) ---
        # Initialised from the symmetrically-normalised anatomical prior.
        # Registered as nn.Parameter so gradients flow through it, allowing
        # the model to refine connectivity beyond the hard-coded skeleton.
        self.adj = nn.Parameter(
            torch.from_numpy(norm_adj_matrix).float(),
            requires_grad=True
        )

        # Per-node linear feature transform (shared MLP via 1×1 conv)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V, M)
        b, c, t, v, m = x.size()

        # Fold M into B for Conv2d compatibility: (B*M, C, T, V)
        x = x.permute(0, 4, 1, 2, 3).contiguous()   # (B, M, C, T, V)
        x = x.view(b * m, c, t, v)

        # Feature transformation: (B*M, C_out, T, V)
        x = self.conv(x)

        # Graph aggregation via right-multiplication by Â:
        #   shape (B*M, C_out, T, V) @ (V, V) → (B*M, C_out, T, V)
        # torch.matmul broadcasts over the leading (B*M, C_out, T) dims.
        x = torch.matmul(x, self.adj)

        # Restore M dimension: (B, C_out, T, V, M)
        x = x.view(b, m, self.out_channels, t, v)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        return x


class TemporalConvolution(nn.Module):
    """
    1-D temporal convolution applied independently per joint.

    Kernel spans `kernel_size` frames and 1 vertex, so it models temporal
    dynamics at each joint without mixing spatial information.

    Input shape:  (B, C_in,  T,        V, M)
    Output shape: (B, C_out, T//stride, V, M)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 9, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2   # 'same' padding along T
        self.t_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, v, m = x.size()

        # Fold M: (B*M, C, T, V)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(b * m, c, t, v)

        x = self.t_conv(x)   # (B*M, C_out, T', V)

        t_out = x.size(2)
        x = x.view(b, m, x.size(1), t_out, v)
        x = x.permute(0, 2, 3, 4, 1).contiguous()   # (B, C_out, T', V, M)

        return x


class ST_GCN_Block(nn.Module):
    """
    One Spatial-Temporal GCN block:
        Spatial GCN  → BN → ReLU → Temporal Conv → BN → ReLU → (+residual)

    BatchNorm2d is applied after both the spatial and temporal convolutions,
    consistent with Yan et al. (2018).  All BN layers operate on the
    (B*M, C, T, V) view for consistency — this also aligns with the residual
    branch's Conv2d which works in the same view.

    Args:
        in_channels  (int): Input feature channels.
        out_channels (int): Output feature channels.
        adj_matrix   (np.ndarray): Normalised adjacency, shape (V, V).
        t_kernel_size(int): Temporal kernel size. Default 9.
        stride       (int): Temporal stride (1 = same length, 2 = halved).
        residual     (bool): Whether to add a residual/skip connection.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 adj_matrix: np.ndarray, t_kernel_size: int = 9,
                 stride: int = 1, residual: bool = True):
        super().__init__()

        # Spatial graph conv + its own BN
        self.gcn = SpatialGraphConvolution(in_channels, out_channels,
                                           adj_matrix)
        self.bn_gcn = nn.BatchNorm2d(out_channels)

        # Temporal conv + its own BN
        self.t_conv = TemporalConvolution(out_channels, out_channels,
                                          kernel_size=t_kernel_size,
                                          stride=stride)
        self.bn_tcn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Residual / skip connection
        if not residual:
            self.residual = None
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            # Project with a strided 1×1 conv + BN when dimensions change
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def _apply_bn2d(self, module: nn.Module,
                    x: torch.Tensor) -> torch.Tensor:
        """
        Helper: apply a BatchNorm2d to a 5D (B, C, T, V, M) tensor by
        temporarily folding M into B, running BN, then unfolding.
        """
        b, c, t, v, m = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(b * m, c, t, v)
        x = module(x)
        x = x.view(b, m, c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Residual branch (computed before modifying x) ---
        res = x

        # --- Main path: Spatial GCN ---
        x = self.gcn(x)                          # (B, C_out, T, V, M)
        x = self._apply_bn2d(self.bn_gcn, x)
        x = self.relu(x)

        # --- Main path: Temporal Conv ---
        x = self.t_conv(x)                       # (B, C_out, T', V, M)
        x = self._apply_bn2d(self.bn_tcn, x)

        # --- Residual branch ---
        if self.residual is not None:
            if isinstance(self.residual, nn.Identity):
                pass   # res unchanged
            else:
                # Project via Conv2d+BN in (B*M, C, T, V) view
                b, c, t, v, m = res.size()
                res = res.permute(0, 4, 1, 2, 3).contiguous()
                res = res.view(b * m, c, t, v)
                res = self.residual(res)         # (B*M, C_out, T', V)
                t_out = res.size(2)
                res = res.view(b, m, res.size(1), t_out, v)
                res = res.permute(0, 2, 3, 4, 1).contiguous()

            x = x + res

        return self.relu(x)