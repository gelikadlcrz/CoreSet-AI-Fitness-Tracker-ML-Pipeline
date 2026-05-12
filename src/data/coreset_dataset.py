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
archives and computes per-frame relative joint angles inline.  The resulting
node feature for joint j is the vector of angles formed by every triplet
(parent, j, child) in the anatomical chain.  For joints with a single
neighbour pair the feature is a scalar angle; for hubs (shoulder, hip) with
multiple neighbours each pair contributes one entry.  A fixed-size feature
vector per node is produced by padding/truncating to `angle_feature_dim`.

Density-map ground truth
-------------------------
Previously calculated from JSON time_points, the density map is now pre-computed 
and stored directly in the .npz archive. The dataset scales it dynamically 
during temporal padding/sampling so the integral (total rep count) is preserved.

Normalisation
-------------
FIX (Issue 4): Z-score normalisation has been removed from __getitem__.
Statistics are computed in train_stgcn.py from the training partition ONLY
and stored as tensors of shape (1, C, 1, 1, 1). They are applied in the
training loop after the DataLoader has stacked samples into batches of shape
(B, C, T, V, 1), where the broadcast axes are unambiguous.

Applying a flat (C,) array inside __getitem__ against angle_seq of shape
(T, C) would broadcast along the time axis instead of the channel axis,
producing incorrect normalisation. The self.mean_std attribute is therefore
deprecated and retained only for backward compatibility; it is never set or
used in the current pipeline.

Noise injection
---------------
FIX (Issue 5): Gaussian noise injection has been removed from __getitem__
and moved to the training loop in train_stgcn.py, where it is applied AFTER
Z-score normalisation. This ensures the perturbation magnitude (±0.05) is
relative to one standardised unit — a controlled fraction of one standard
deviation — rather than being proportional to the raw angle magnitude (which
varies widely across different joint types, e.g. 5°–170°).
"""

import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Joint-angle computation helpers
# ---------------------------------------------------------------------------

# Anatomical triplets (proximal_joint, vertex_joint, distal_joint) for which
# the bond angle is computed.  Each triplet yields one angle value per frame.
# Ordered to produce a consistent, fixed-size feature vector.
ANGLE_TRIPLETS = [
   # Trunk
   (11, 23, 25),   # left shoulder–hip–knee (trunk lean)
   (12, 24, 26),   # right shoulder–hip–knee

   # Left arm
   (11, 13, 15),   # left shoulder–elbow–wrist (elbow flexion)
   (13, 15, 17),   # left elbow–wrist–pinky

   # Right arm
   (12, 14, 16),   # right shoulder–elbow–wrist
   (14, 16, 18),   # right elbow–wrist–pinky

   # Left leg
   (23, 25, 27),   # left hip–knee–ankle (knee flexion)
   (25, 27, 29),   # left knee–ankle–heel

   # Right leg
   (24, 26, 28),   # right hip–knee–ankle
   (26, 28, 30),   # right knee–ankle–heel

   # Shoulder-hip-knee in sagittal plane (hip flexion proxy)
   (11, 23, 24),   # left shoulder–left hip–right hip
   (12, 24, 23),   # right shoulder–right hip–left hip

   # Elbow-shoulder-hip (shoulder flexion/abduction proxy)
   (13, 11, 23),   # left elbow–left shoulder–left hip
   (14, 12, 24),   # right elbow–right shoulder–right hip
]

ANGLE_FEATURE_DIM = len(ANGLE_TRIPLETS)


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
   """
   Returns the angle at vertex b (in radians) formed by rays b→a and b→c.

   Uses the dot-product formula:
       θ = arccos( (ba · bc) / (|ba| · |bc|) )

   Clipped to [-1, 1] before arccos to guard against floating-point errors.
   Returns 0.0 if either ray has zero length (degenerate / occluded joint).
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
       coords (np.ndarray): shape (33, 4) — world (x, y, z, v) per landmark.

   Returns:
       angles (np.ndarray): shape (ANGLE_FEATURE_DIM,) — one angle per
                            triplet, in radians.
   """
   angles = np.zeros(ANGLE_FEATURE_DIM, dtype=np.float32)
   for idx, (i, j, k) in enumerate(ANGLE_TRIPLETS):
       # Force strictly X,Y,Z into the math so visibility score is ignored
       angles[idx] = _angle_between(coords[i][:3], coords[j][:3], coords[k][:3])
   return angles


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CoreSetGCN_Dataset(Dataset):
   EXERCISE_MAP = {
       'squat':       0,
       'push_up':     1,
       'bench_press': 2,
       'pull_up':     3,
   }

   def __init__(
       self,
       data_dir: str,
       split_file: str,
       split_type: str = 'train',
       max_frames: int = 150,
       augment: bool = False,
   ):
       self.data_dir   = Path(data_dir)
       self.max_frames = max_frames
       self.augment    = augment

       # FIX (Issue 4): mean_std is deprecated — retained only for backward
       # compatibility with any external code that may reference it.
       # It is never set or consulted in the current training pipeline.
       # Normalisation is performed exclusively in train_stgcn.py.
       self.mean_std: Optional[tuple] = None

       self.file_paths: List[Path] = []
       self.labels: List[int] = []

       # Load the centralized split configuration
       with open(split_file, 'r') as f:
           splits = json.load(f)

       if split_type not in splits:
           raise ValueError(
               f"split_type must be one of "
               f"{list(splits.keys() - {'metadata', 'subjects'})}"
           )

       # Get the list of relative file paths for this split
       split_files = splits[split_type]

       for rel_path in split_files:
           full_path = self.data_dir / rel_path
           self.file_paths.append(full_path)

           # Extract the exercise class from the parent directory name
           exercise_name = Path(rel_path).parent.name
           if exercise_name in self.EXERCISE_MAP:
               self.labels.append(self.EXERCISE_MAP[exercise_name])
           else:
               raise ValueError(
                   f"Unknown exercise class directory: {exercise_name}"
               )

   def __len__(self) -> int:
       return len(self.file_paths)

   # ------------------------------------------------------------------
   # Augmentation
   # ------------------------------------------------------------------

   def _apply_augmentations(self, coords_seq: np.ndarray) -> np.ndarray:
       """
       Apply spatial mirroring augmentation to the raw coordinate sequence
       before angle computation.

       NOTE: Gaussian noise is NOT applied here (Issue 5 fix). It is applied
       in train_stgcn.py AFTER Z-score normalisation so that the perturbation
       scale is consistent across all angle channels.
       """
       if np.random.rand() > 0.5:
           coords_seq = coords_seq.copy()
           coords_seq[:, :, 0] *= -1.0   # invert X axis

           # BlazePose left-right index pairs (body landmarks only)
           left_nodes  = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
           right_nodes = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
           tmp = coords_seq[:, left_nodes, :].copy()
           coords_seq[:, left_nodes, :]  = coords_seq[:, right_nodes, :]
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
           Spatiotemporal graph of raw (un-normalised) joint angles.
           C = ANGLE_FEATURE_DIM (14), T = max_frames, V = 33 nodes.
           Z-score normalisation is applied in the training loop, not here.
       label       : int
           Integer class index (0–3).
       density_gt  : torch.Tensor, shape (T,)
           Normalised Gaussian density map over the temporal window.
       """
       filepath = self.file_paths[idx]
       
       # 1. Load the NPZ archive
       data = np.load(filepath)
       coords_seq = data['pose_sequence']  # Shape: (T, 33, 4)
       density_gt = data['density']        # Shape: (T,)
       
       # Prefer the category_index baked into the npz, fallback to folder mapping
       if 'category_index' in data.files:
           label = int(data['category_index'])
       else:
           label = self.labels[idx]
           
       actual_frames = coords_seq.shape[0]

       # --- Augmentation on raw coordinates (before angle computation) ---
       if self.augment:
           coords_seq = self._apply_augmentations(coords_seq)

       # --- Temporal sampling / padding ---
       if self.augment and actual_frames > self.max_frames:
           idx_seq = np.sort(np.random.choice(actual_frames, self.max_frames, replace=False))
           coords_seq = coords_seq[idx_seq]
           density_gt = density_gt[idx_seq]
       elif actual_frames >= self.max_frames:
           idx_seq = np.linspace(0, actual_frames - 1, self.max_frames, dtype=int)
           coords_seq = coords_seq[idx_seq]
           density_gt = density_gt[idx_seq]
       else:
           # PAD FIX: Repeat last frame instead of collapsing to origin
           if actual_frames > 0:
               pad_length = self.max_frames - actual_frames
               pad_coords = np.tile(coords_seq[-1:], (pad_length, 1, 1))
               coords_seq = np.concatenate([coords_seq, pad_coords], axis=0)
               
               # Pad density map with zeros
               pad_density = np.zeros(pad_length, dtype=np.float32)
               density_gt = np.concatenate([density_gt, pad_density], axis=0)
           else:
               coords_seq = np.zeros((self.max_frames, 33, 4), dtype=np.float32)
               density_gt = np.zeros(self.max_frames, dtype=np.float32)

       # Density Integral Preservation: Scaling the density map ensures that 
       # temporal skipping/padding doesn't alter the total rep count (equivalent 
       # to the integral guarantee in the old _get_ground_truth_density_map).
       target_count = float(data['count']) if 'count' in data.files else float(density_gt.sum())
       current_sum = density_gt.sum()
       if current_sum > 0:
           density_gt = (density_gt / current_sum) * target_count

       # --- Compute joint angles: (max_frames, ANGLE_FEATURE_DIM) ---
       angle_seq = np.stack(
           [compute_joint_angles(coords_seq[t]) for t in range(self.max_frames)],
           axis=0
       )  # shape: (max_frames, ANGLE_FEATURE_DIM)

       # FIX (Issue 5): Gaussian noise removed from here.
       # FIX (Issue 4): Z-score normalisation removed from here.

       # --- TOPOLOGY FIX: Map angles strictly to anatomical vertices ---
       # Output shape: (Channels, Frames, Vertices, Persons) → (C, T, V, 1)
       # where C = ANGLE_FEATURE_DIM, T = max_frames, V = 33
       feature_tensor = np.zeros(
           (ANGLE_FEATURE_DIM, self.max_frames, 33, 1), dtype=np.float32
       )

       for idx_angle, (i, j, k) in enumerate(ANGLE_TRIPLETS):
           # 'j' is the vertex joint (e.g., elbow in shoulder–elbow–wrist)
           feature_tensor[idx_angle, :, j, 0] = angle_seq[:, idx_angle]

       return torch.from_numpy(feature_tensor), label, torch.from_numpy(density_gt)