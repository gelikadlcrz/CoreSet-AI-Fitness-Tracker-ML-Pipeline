"""
metrics.py — Evaluation metrics for the CoreSet ST-GCN framework.

Implements the counting and classification metrics used by the ST-GCN
training/evaluation scripts.

Counting note
-------------
The ST-GCN density head predicts a normalized peak-probability curve, not an
absolute density integral. Rep count should therefore be recovered by peak
finding with the same adaptive post-processing used during validation.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import torch


class CoreSetEvaluator:
    """
    Stateless evaluator.  All methods accept plain lists or numpy arrays
    (for count metrics) or torch Tensors (for classification accuracy).
    """

    def __init__(self, exercise_classes: List[str] = None):
        if exercise_classes is None:
            exercise_classes = ['squat', 'push_up', 'bench_press', 'pull_up']
        self.classes = exercise_classes
        self.num_classes = len(exercise_classes)

    # ------------------------------------------------------------------
    # Counting metrics
    # ------------------------------------------------------------------

    def calculate_normalized_mae(self,
                                  predictions: List[float],
                                  ground_truths: List[float]) -> float:
        """
        Normalized Mean Absolute Error:
            mean(|pred - gt| / gt) for gt > 0
        """
        predictions = np.asarray(predictions, dtype=float)
        ground_truths = np.asarray(ground_truths, dtype=float)

        df = pd.DataFrame({'pred': predictions, 'gt': ground_truths})
        mask = df['gt'] > 0
        abs_err = np.abs(df['pred'] - df['gt'])

        df['normalized_ae'] = 0.0
        df.loc[mask, 'normalized_ae'] = abs_err[mask] / df.loc[mask, 'gt']

        return float(df['normalized_ae'].mean())

    def calculate_rmse(self,
                       predictions: List[float],
                       ground_truths: List[float]) -> float:
        """Root Mean Square Error on raw repetition counts."""
        predictions = np.asarray(predictions, dtype=float)
        ground_truths = np.asarray(ground_truths, dtype=float)
        return float(np.sqrt(np.mean((predictions - ground_truths) ** 2)))

    def calculate_obo_accuracy(self,
                                predictions: List[float],
                                ground_truths: List[float],
                                tolerance: int = 1) -> float:
        """Off-By-One accuracy: |pred - gt| <= tolerance."""
        predictions = np.asarray(predictions, dtype=float)
        ground_truths = np.asarray(ground_truths, dtype=float)
        correct = (np.abs(predictions - ground_truths) <= tolerance).sum()
        return float(correct) / max(len(predictions), 1)

    # ------------------------------------------------------------------
    # Classification metric
    # ------------------------------------------------------------------

    def calculate_classification_accuracy(self,
                                           logits: torch.Tensor,
                                           ground_truths: torch.Tensor) -> float:
        """Top-1 classification accuracy."""
        probabilities = torch.softmax(logits, dim=1)
        predicted = probabilities.argmax(dim=1)
        correct = (predicted == ground_truths).sum().item()
        return correct / max(len(ground_truths), 1)

    # ------------------------------------------------------------------
    # Density-map → rep count conversion
    # ------------------------------------------------------------------

    @staticmethod
    def density_map_to_count(
        density_map: np.ndarray,
        threshold: Optional[float] = None,
        min_distance: int = 10,
        adaptive: bool = True,
        threshold_ratio: float = 0.30,
        threshold_floor: float = 0.15,
        prominence_ratio: float = 0.05,
    ) -> int:
        """
        Convert a predicted normalized density map to a discrete rep count.

        The previous fixed threshold=0.5 could under-count when the model's
        peak heights were valid but below 0.5. The training loop used an
        adaptive threshold during validation, so evaluation should follow the
        same rule unless explicitly overridden.

        Args:
            density_map: shape (T,), predicted values in [0, 1].
            threshold: fixed threshold. If None and adaptive=True, the threshold
                is max(max(density_map) * threshold_ratio, threshold_floor).
            min_distance: minimum frames between detected peaks.
            adaptive: enable adaptive thresholding.
            threshold_ratio: fraction of the sample's max peak height.
            threshold_floor: minimum allowed threshold.
            prominence_ratio: minimum prominence as a fraction of the sample's
                max value. Helps suppress tiny noisy bumps.
        """
        from scipy.signal import find_peaks

        density_map = np.asarray(density_map, dtype=np.float32).reshape(-1)
        if density_map.size == 0:
            return 0

        max_val = float(np.nanmax(density_map))
        if not np.isfinite(max_val) or max_val <= 0:
            return 0

        if threshold is None:
            if adaptive:
                threshold = max(max_val * float(threshold_ratio), float(threshold_floor))
            else:
                threshold = 0.5

        # If the learned peak heights are all below the floor, still allow the
        # strongest peaks to be detected instead of forcing zero reps.
        threshold = min(float(threshold), max_val * 0.95)

        prominence = max(max_val * float(prominence_ratio), 1e-6)
        peaks, _ = find_peaks(
            density_map,
            height=threshold,
            distance=int(min_distance),
            prominence=prominence,
        )
        return int(len(peaks))
