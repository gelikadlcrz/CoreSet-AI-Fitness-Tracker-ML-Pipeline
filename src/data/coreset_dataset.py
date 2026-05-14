"""
coreset_dataset.py — PyTorch Dataset for the CoreSet ST-GCN framework.

Node feature representation
---------------------------
The methodology specifies that *relative joint angles*, not raw XYZ world
coordinates, serve as node attributes:

   "From these world-coordinate positions, relative joint angles are computed
    for each anatomically connected joint pair (e.g., elbow flexion angle
    from shoulder, elbow, and wrist landmarks). These angular values, rather
    than raw XYZ coordinates, serve as the node attributes within the graph.
    This representation is inherently more viewpoint-invariant than absolute
    positional coordinates, as joint angles remain stable under camera
    rotation and translation." — Methodology, Data Preprocessing

This dataset reads the raw (x, y, z) world coordinates stored in the NPZ
archives and computes per-frame relative joint angles inline. The resulting
node feature for joint j is the vector of angles formed by every triplet
(parent, j, child) in the anatomical chain.

Density-map ground truth
-------------------------
Newer NPZ archives contain a pre-computed density map. For older/legacy NPZ
archives that do not contain 'density', this dataset creates a fallback density
map from the saved repetition count so ST-GCN training remains compatible.

Normalisation
-------------
Z-score normalisation has been removed from __getitem__. Statistics are
computed in train_stgcn.py from the training partition only and applied in the
training loop after the DataLoader stacks samples into batches.

Noise injection
---------------
Gaussian noise injection has been removed from __getitem__ and moved to the
training loop in train_stgcn.py, where it is applied after Z-score
normalisation.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Joint-angle computation helpers
# ---------------------------------------------------------------------------

ANGLE_TRIPLETS = [
    # Trunk
    (11, 23, 25),   # left shoulder–hip–knee
    (12, 24, 26),   # right shoulder–hip–knee

    # Left arm
    (11, 13, 15),   # left shoulder–elbow–wrist
    (13, 15, 17),   # left elbow–wrist–pinky

    # Right arm
    (12, 14, 16),   # right shoulder–elbow–wrist
    (14, 16, 18),   # right elbow–wrist–pinky

    # Left leg
    (23, 25, 27),   # left hip–knee–ankle
    (25, 27, 29),   # left knee–ankle–heel

    # Right leg
    (24, 26, 28),   # right hip–knee–ankle
    (26, 28, 30),   # right knee–ankle–heel

    # Hip/trunk proxies
    (11, 23, 24),   # left shoulder–left hip–right hip
    (12, 24, 23),   # right shoulder–right hip–left hip

    # Shoulder proxies
    (13, 11, 23),   # left elbow–left shoulder–left hip
    (14, 12, 24),   # right elbow–right shoulder–right hip
]

ANGLE_FEATURE_DIM = len(ANGLE_TRIPLETS)


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Returns the angle at vertex b, in radians, formed by rays b→a and b→c.
    """
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return 0.0

    cos_theta = np.dot(ba, bc) / (norm_ba * norm_bc)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def compute_joint_angles(coords: np.ndarray) -> np.ndarray:
    """
    Compute relative joint angles for one frame.

    Args:
        coords: shape (33, 4), world x/y/z/visibility per landmark.

    Returns:
        angles: shape (ANGLE_FEATURE_DIM,), one angle per triplet in radians.
    """
    angles = np.zeros(ANGLE_FEATURE_DIM, dtype=np.float32)

    for idx, (i, j, k) in enumerate(ANGLE_TRIPLETS):
        angles[idx] = _angle_between(
            coords[i][:3],
            coords[j][:3],
            coords[k][:3]
        )

    return angles


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CoreSetGCN_Dataset(Dataset):
    EXERCISE_MAP = {
        "squat": 0,
        "push_up": 1,
        "bench_press": 2,
        "pull_up": 3,
    }

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        split_type: str = "train",
        max_frames: int = 150,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.augment = augment

        # Deprecated, retained only for backward compatibility.
        self.mean_std: Optional[tuple] = None

        self.file_paths: List[Path] = []
        self.labels: List[int] = []

        with open(split_file, "r") as f:
            splits = json.load(f)

        if split_type not in splits:
            valid_splits = [k for k in splits.keys() if k not in {"metadata", "subjects"}]
            raise ValueError(f"split_type must be one of {valid_splits}")

        split_files = splits[split_type]

        for rel_path in split_files:
            full_path = self.data_dir / rel_path
            self.file_paths.append(full_path)

            exercise_name = Path(rel_path).parent.name

            if exercise_name in self.EXERCISE_MAP:
                self.labels.append(self.EXERCISE_MAP[exercise_name])
            else:
                raise ValueError(f"Unknown exercise class directory: {exercise_name}")

    def __len__(self) -> int:
        return len(self.file_paths)

    # ------------------------------------------------------------------
    # Density fallback for older NPZ files
    # ------------------------------------------------------------------

    def _make_fallback_density(self, num_frames: int, rep_count: float) -> np.ndarray:
        """
        Builds a fallback density map for legacy .npz files that do not contain
        a precomputed 'density' array.

        The density sum is normalized to the repetition count.
        """
        num_frames = int(num_frames)
        rep_count = float(rep_count)

        if num_frames <= 0:
            return np.zeros(1, dtype=np.float32)

        if rep_count <= 0:
            return np.zeros(num_frames, dtype=np.float32)

        density = np.zeros(num_frames, dtype=np.float32)

        peak_count = max(1, int(round(rep_count)))
        peak_positions = np.linspace(
            0,
            num_frames - 1,
            peak_count + 2,
            dtype=np.float32
        )[1:-1]

        sigma = max(2.0, num_frames / max(peak_count * 6.0, 1.0))
        frame_ids = np.arange(num_frames, dtype=np.float32)

        for center in peak_positions:
            density += np.exp(-0.5 * ((frame_ids - center) / sigma) ** 2)

        density_sum = float(density.sum())
        if density_sum > 0:
            density = density / density_sum * rep_count

        return density.astype(np.float32)

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _apply_augmentations(self, coords_seq: np.ndarray) -> np.ndarray:
        """
        Apply spatial mirroring augmentation to the raw coordinate sequence
        before angle computation.
        """
        if np.random.rand() > 0.5:
            coords_seq = coords_seq.copy()
            coords_seq[:, :, 0] *= -1.0

            left_nodes = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
            right_nodes = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

            tmp = coords_seq[:, left_nodes, :].copy()
            coords_seq[:, left_nodes, :] = coords_seq[:, right_nodes, :]
            coords_seq[:, right_nodes, :] = tmp

        return coords_seq

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        data_tensor : torch.Tensor, shape (C, T, V, 1)
            Spatiotemporal graph of raw un-normalised joint angles.

        label : int
            Integer class index.

        density_gt : torch.Tensor, shape (T,)
            Density map over the temporal window.
        """
        filepath = self.file_paths[idx]

        data = np.load(filepath)

        coords_seq = data["pose_sequence"].astype(np.float32)  # Shape: (T, 33, 4)
        actual_frames = coords_seq.shape[0]

        # Compatible with both newer NPZ files and older legacy NPZ files.
        if "density" in data.files:
            density_gt = data["density"].astype(np.float32)
        else:
            rep_count = float(data["count"]) if "count" in data.files else 0.0
            density_gt = self._make_fallback_density(actual_frames, rep_count)

        # Prefer category_index from NPZ, fallback to folder mapping.
        if "category_index" in data.files:
            label = int(data["category_index"])
        else:
            label = self.labels[idx]

        # Augmentation on raw coordinates before angle computation.
        if self.augment:
            coords_seq = self._apply_augmentations(coords_seq)

        # Temporal sampling / padding.
        if self.augment and actual_frames > self.max_frames:
            idx_seq = np.sort(
                np.random.choice(actual_frames, self.max_frames, replace=False)
            )
            coords_seq = coords_seq[idx_seq]
            density_gt = density_gt[idx_seq]

        elif actual_frames >= self.max_frames:
            idx_seq = np.linspace(0, actual_frames - 1, self.max_frames, dtype=int)
            coords_seq = coords_seq[idx_seq]
            density_gt = density_gt[idx_seq]

        else:
            if actual_frames > 0:
                pad_length = self.max_frames - actual_frames

                pad_coords = np.tile(coords_seq[-1:], (pad_length, 1, 1))
                coords_seq = np.concatenate([coords_seq, pad_coords], axis=0)

                pad_density = np.zeros(pad_length, dtype=np.float32)
                density_gt = np.concatenate([density_gt, pad_density], axis=0)
            else:
                coords_seq = np.zeros((self.max_frames, 33, 4), dtype=np.float32)
                density_gt = np.zeros(self.max_frames, dtype=np.float32)

        # Preserve the density integral so temporal sampling/padding does not
        # change the total repetition count.
        if "count" in data.files:
            target_count = float(data["count"])
        else:
            target_count = float(density_gt.sum())

        current_sum = float(density_gt.sum())
        if current_sum > 0:
            density_gt = (density_gt / current_sum) * target_count

        density_gt = density_gt.astype(np.float32)

        # Compute joint angles: (max_frames, ANGLE_FEATURE_DIM)
        angle_seq = np.stack(
            [compute_joint_angles(coords_seq[t]) for t in range(self.max_frames)],
            axis=0
        )

        # Map angle features to their anatomical vertex joints.
        # Output shape: (C, T, V, 1)
        feature_tensor = np.zeros(
            (ANGLE_FEATURE_DIM, self.max_frames, 33, 1),
            dtype=np.float32
        )

        for idx_angle, (_, vertex_joint, _) in enumerate(ANGLE_TRIPLETS):
            feature_tensor[idx_angle, :, vertex_joint, 0] = angle_seq[:, idx_angle]

        return (
            torch.from_numpy(feature_tensor),
            label,
            torch.from_numpy(density_gt)
        )