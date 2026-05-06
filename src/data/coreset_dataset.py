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

This dataset reads the raw (x, y, z) world coordinates stored in the JSON
files and computes per-frame relative joint angles inline.  The resulting
node feature for joint j is the vector of angles formed by every triplet
(parent, j, child) in the anatomical chain.  For joints with a single
neighbour pair the feature is a scalar angle; for hubs (shoulder, hip) with
multiple neighbours each pair contributes one entry.  A fixed-size feature
vector per node is produced by padding/truncating to `angle_feature_dim`.

Density-map ground truth
-------------------------
The JSON files are expected to contain a `time_points` key whose value is a
list of integer frame indices at which a repetition *completes* (i.e. the
peak or valley of the movement cycle, depending on annotation convention).
Each point becomes a normalized Gaussian bump.
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
       coords (np.ndarray): shape (33, 3) — world (x, y, z) per landmark.

   Returns:
       angles (np.ndarray): shape (ANGLE_FEATURE_DIM,) — one angle per
                            triplet, in radians, normalised to [0, 1] by
                            dividing by π.
   """
   angles = np.zeros(ANGLE_FEATURE_DIM, dtype=np.float32)
   for idx, (i, j, k) in enumerate(ANGLE_TRIPLETS):
       angles[idx] = _angle_between(coords[i], coords[j], coords[k]) / math.pi
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

   def __init__(self, data_dir: str, split_file: str, split_type: str = 'train', max_frames: int = 150, augment: bool = False):
       self.data_dir = Path(data_dir)
       self.max_frames = max_frames
       self.augment = augment

       self.file_paths: List[Path] = []
       self.labels: List[int] = []
       
       # Load the centralized split configuration
       with open(split_file, 'r') as f:
           splits = json.load(f)
           
       if split_type not in splits:
           raise ValueError(f"split_type must be one of {list(splits.keys() - {'metadata', 'subjects'})}")
           
       # Get the list of relative file paths for this split (e.g., ["bench_press/bench_press_094.json", ...])
       split_files = splits[split_type]
       
       for rel_path in split_files:
           full_path = self.data_dir / rel_path
           self.file_paths.append(full_path)
           
           # Extract the exercise class from the parent directory name
           exercise_name = Path(rel_path).parent.name
           if exercise_name in self.EXERCISE_MAP:
               self.labels.append(self.EXERCISE_MAP[exercise_name])
           else:
               raise ValueError(f"Unknown exercise class directory: {exercise_name}")

   def __len__(self) -> int:
       return len(self.file_paths)

   # ------------------------------------------------------------------
   # Augmentation
   # ------------------------------------------------------------------

   def _apply_augmentations(self, coords_seq: np.ndarray) -> np.ndarray:
       """
       Apply in-place augmentations to the raw coordinate sequence before
       angle computation.
       """
       if np.random.rand() > 0.5:
           noise = np.random.normal(0.0, 0.005, coords_seq.shape)
           coords_seq = coords_seq + noise.astype(np.float32)

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
   # Data loading and shaping
   # ------------------------------------------------------------------

   def _load_and_shape_data(self, filepath: Path) -> torch.Tensor:
       with open(filepath, 'r') as f:
           data = json.load(f)

       # --- JSON LIST FIX ---
       if isinstance(data, list):
           frames_list = data
       elif isinstance(data, dict):
           frames_list = data.get('frames', [])
       else:
           frames_list = []

       actual_frames = len(frames_list)

       # --- Extract raw (x, y, z) world coordinates: shape (T, 33, 3) ---
       coords_seq = np.zeros((actual_frames, 33, 3), dtype=np.float32)
       for t_idx, frame in enumerate(frames_list):
           # Handle new format gracefully
           if isinstance(frame, dict):
               lm = frame.get('landmarks', [])
           elif isinstance(frame, list):
               lm = frame
           else:
               continue
               
           if lm:
            coords = []

            for point in lm[:33]:
                if isinstance(point, dict):
                    coords.append([
                        point.get("x", 0.0),
                        point.get("y", 0.0),
                        point.get("z", 0.0)
                    ])
                else:
                    coords.append(point[:3])

            arr = np.array(coords, dtype=np.float32)

            if arr.shape == (33, 3):
                coords_seq[t_idx] = arr

       # --- Augmentation on raw coordinates (before angle computation) ---
       if self.augment:
           coords_seq = self._apply_augmentations(coords_seq)

       # --- Temporal sampling / padding ---
       if self.augment and actual_frames > self.max_frames:
           idx = np.sort(np.random.choice(actual_frames, self.max_frames, replace=False))
           coords_seq = coords_seq[idx]
       elif actual_frames >= self.max_frames:
           idx = np.linspace(0, actual_frames - 1, self.max_frames, dtype=int)
           coords_seq = coords_seq[idx]
       else:
           pad = np.zeros((self.max_frames - actual_frames, 33, 3), dtype=np.float32)
           coords_seq = np.concatenate([coords_seq, pad], axis=0)

       # --- Compute joint angles: (max_frames, ANGLE_FEATURE_DIM) ---
       angle_seq = np.stack(
           [compute_joint_angles(coords_seq[t]) for t in range(self.max_frames)],
           axis=0
       )  # shape: (max_frames, ANGLE_FEATURE_DIM)

       # --- Build node feature tensor: broadcast angles to all nodes ---
       angle_seq = angle_seq[:, np.newaxis, :]          # (T, 1, C)
       angle_seq = np.tile(angle_seq, (1, 33, 1))       # (T, V, C)
       angle_seq = angle_seq.transpose(2, 0, 1)         # (C, T, V)
       angle_seq = angle_seq[:, :, :, np.newaxis]       # (C, T, V, 1)

       return torch.from_numpy(angle_seq)

   # ------------------------------------------------------------------
   # Density-map ground truth
   # ------------------------------------------------------------------

   def _get_ground_truth_density_map(self, filepath: Path, actual_frames: int) -> torch.Tensor:
       """
       Build normalized density map from time_points.

       Current labeled_json format:
           "time_points": [peak1, peak2, peak3, ...]
           "rep_count": number of detected repetitions

       Each time point represents one detected repetition peak/valley.
       Each point becomes a normalized Gaussian bump.
       The density map sum should be approximately equal to rep_count.
       """

       density_gt = np.zeros(self.max_frames, dtype=np.float32)

       with open(filepath, "r") as f:
           data = json.load(f)

       if not isinstance(data, dict):
           return torch.from_numpy(density_gt)

       time_points = data.get("time_points", [])
       rep_count = data.get("rep_count", len(time_points))

       if not time_points or actual_frames <= 0:
           return torch.from_numpy(density_gt)

       sigma = 2.0

       for point in time_points:
           point = int(point)

           if actual_frames >= self.max_frames:
               mapped = int((point / actual_frames) * self.max_frames)
           else:
               mapped = point

           mapped = max(0, min(mapped, self.max_frames - 1))

           for t in range(self.max_frames):
               density_gt[t] += (
                   np.exp(-((t - mapped) ** 2) / (2 * sigma ** 2))
                   / (sigma * np.sqrt(2 * np.pi))
               )

       total = density_gt.sum()

       if total > 0:
           density_gt = density_gt / total * rep_count

       return torch.from_numpy(density_gt)
   
   # ------------------------------------------------------------------
   # __getitem__
   # ------------------------------------------------------------------

   def __getitem__(self, idx: int):
       filepath = self.file_paths[idx]
       label = self.labels[idx]

       # Safe lightweight pre-read for actual frame count
       with open(filepath, 'r') as f:
           meta = json.load(f)
           
       if isinstance(meta, list):
           actual_frames = len(meta)
       elif isinstance(meta, dict):
           actual_frames = len(meta.get('frames', []))
       else:
           actual_frames = 0

       data_tensor = self._load_and_shape_data(filepath)
       density_gt  = self._get_ground_truth_density_map(filepath, actual_frames)

       return data_tensor, label, density_gt