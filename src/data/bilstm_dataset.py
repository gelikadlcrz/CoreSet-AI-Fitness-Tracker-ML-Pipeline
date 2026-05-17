import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BiLSTMDataset(Dataset):
    """
    Dataset for the BiLSTM baseline model.

    Loads pre-standardised flattened joint-angle tensors produced by the
    harmonisation pipeline.  Because every sequence has already been
    normalised to zero mean / unit variance AND padded/truncated to a
    fixed temporal window during pre-processing, NO further padding or
    truncation is performed here.
    """

    def __init__(
        self,
        data_dir: str,
        split_list: list,
        augment: bool = False,
        cache: bool = True,
    ):
        self.data_dir = data_dir
        self.augment = augment
        self.cache = cache
        self._cache: dict = {}

        self.class_names = ['bench_press', 'pull_up', 'push_up', 'squat']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        self.samples: list = []
        for item in split_list:
            pt_file = item.replace(".npz", ".pt")
            class_name = os.path.dirname(pt_file)
            full_path = os.path.join(data_dir, pt_file)

            if os.path.exists(full_path):
                label = self.class_to_idx[class_name]
                self.samples.append((full_path, label))
            else:
                print(f"Warning: Missing file {full_path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> dict:
        if self.cache and path in self._cache:
            return self._cache[path]
        loaded = torch.load(path, weights_only=False)
        if self.cache:
            self._cache[path] = loaded
        return loaded

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        loaded = self._load(path)

        # Data is already pre-standardised; use it directly.
        data = loaded['sequence'].float()           # (T, input_size)
        rep_count = float(loaded.get('rep_count', 0))

        if self.augment:
            data = self._augment(data)

        return data, label, rep_count

    # ------------------------------------------------------------------
    # Augmentation (training only)
    # ------------------------------------------------------------------

    def _augment(self, data: torch.Tensor) -> torch.Tensor:
        """Light stochastic augmentation that preserves sequence length."""
        num_frames = data.shape[0]

        # Random temporal crop (keep 80-100 % of frames, then resize back)
        if num_frames > 10 and torch.rand(1).item() < 0.8:
            keep_ratio = torch.empty(1).uniform_(0.8, 1.0).item()
            keep_frames = max(10, int(num_frames * keep_ratio))
            start = torch.randint(0, num_frames - keep_frames + 1, (1,)).item()
            cropped = data[start: start + keep_frames]           # (T', C)
            # Resize back to original length so batches remain uniform
            data = (
                F.interpolate(
                    cropped.T.unsqueeze(0),                      # (1, C, T')
                    size=num_frames,
                    mode='linear',
                    align_corners=False,
                )
                .squeeze(0)
                .T                                               # (T, C)
            )

        # Speed perturbation: subtle temporal warp then resize back
        if num_frames > 5 and torch.rand(1).item() < 0.3:
            scale = torch.empty(1).uniform_(0.9, 1.1).item()
            warped_len = max(5, int(num_frames * scale))
            data = (
                F.interpolate(
                    data.T.unsqueeze(0),
                    size=warped_len,
                    mode='linear',
                    align_corners=False,
                )
                .squeeze(0)
                .T
            )
            # Resize back to original length
            data = (
                F.interpolate(
                    data.T.unsqueeze(0),
                    size=num_frames,
                    mode='linear',
                    align_corners=False,
                )
                .squeeze(0)
                .T
            )

        # Small additive Gaussian noise
        if torch.rand(1).item() < 0.5:
            data = data + torch.randn_like(data) * 0.01

        return data