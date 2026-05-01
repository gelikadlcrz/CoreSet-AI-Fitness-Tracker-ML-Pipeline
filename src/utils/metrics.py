"""
metrics.py — Evaluation metrics for the CoreSet ST-GCN framework.

Implements the three counting metrics and one classification metric specified
in the Validation & Testing section of the methodology:

  Counting metrics (applied to density-map → rep-count conversion):
    1. Normalized MAE  — L1 fidelity, robust to outliers (Izadi et al., 2022)
    2. RMSE            — L2 penalty, flags catastrophic failures
    3. OBO Accuracy    — off-by-one tolerance (Hu et al., 2022 / RepCount)

  Classification metric:
    4. Top-1 Accuracy  — used during training validation loop

  Note on AUC:
    AUC (periodicity detection, following RepNet / Dwibedi et al., 2020) is
    computed externally using sklearn.metrics.roc_auc_score on the raw
    sigmoid density map outputs vs binary ground-truth density maps.
    It is not included here to avoid adding sklearn as a hard dependency on
    the hot path; add it to your test script with:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(density_gt_flat, density_pred_flat)
"""

import numpy as np
import torch
import pandas as pd
from typing import List


class CoreSetEvaluator:
    """
    Stateless evaluator.  All methods accept plain lists or numpy arrays
    (for count metrics) or torch Tensors (for classification accuracy).
    """

    def __init__(self,
                 exercise_classes: List[str] = None):
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
        Normalized Mean Absolute Error (MAE), following the RepCount benchmark
        protocol (Hu et al., 2022):

            MAE = mean( |pred - gt| / gt )   for gt > 0

        Samples with gt == 0 are excluded from the denominator to avoid
        division by zero, as per the benchmark convention.

        Args:
            predictions   : Predicted repetition counts.
            ground_truths : Ground-truth repetition counts.

        Returns:
            Scalar normalized MAE.  Lower is better.
        """
        predictions   = np.asarray(predictions, dtype=float)
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
        """
        Root Mean Square Error on raw (non-normalized) repetition counts.

        RMSE squares errors before averaging, disproportionately penalising
        large miscalculations (e.g. hallucinated reps from tracking failures).
        Used alongside MAE to characterise the error distribution
        (Izadi et al., 2022).

        Returns:
            Scalar RMSE.  Lower is better.
        """
        predictions   = np.asarray(predictions, dtype=float)
        ground_truths = np.asarray(ground_truths, dtype=float)
        return float(np.sqrt(np.mean((predictions - ground_truths) ** 2)))

    def calculate_obo_accuracy(self,
                                predictions: List[float],
                                ground_truths: List[float],
                                tolerance: int = 1) -> float:
        """
        Off-By-One (OBO) accuracy — fraction of videos where the predicted
        count falls within `tolerance` repetitions of the ground truth
        (Hu et al., 2022).

        A higher OBO score is better.  Tolerance=1 is the standard setting.

        Returns:
            Scalar in [0, 1].
        """
        predictions   = np.asarray(predictions, dtype=float)
        ground_truths = np.asarray(ground_truths, dtype=float)
        correct = (np.abs(predictions - ground_truths) <= tolerance).sum()
        return float(correct) / len(predictions)

    # ------------------------------------------------------------------
    # Classification metric
    # ------------------------------------------------------------------

    def calculate_classification_accuracy(self,
                                           logits: torch.Tensor,
                                           ground_truths: torch.Tensor
                                           ) -> float:
        """
        Top-1 classification accuracy.

        Args:
            logits        (Tensor): shape (N, num_classes), raw logits.
            ground_truths (Tensor): shape (N,), integer class indices.

        Returns:
            Scalar accuracy in [0, 1].
        """
        probabilities = torch.softmax(logits, dim=1)
        predicted = probabilities.argmax(dim=1)
        correct   = (predicted == ground_truths).sum().item()
        return correct / len(ground_truths)

    # ------------------------------------------------------------------
    # Density-map → rep count conversion (peak detection)
    # ------------------------------------------------------------------

    @staticmethod
    def density_map_to_count(density_map: np.ndarray,
                             threshold: float = 0.5,
                             min_distance: int = 10) -> int:
        """
        Convert a predicted density map to a discrete repetition count using
        simple threshold-based peak detection.

        A frame is counted as a repetition completion if:
          (a) its sigmoid value exceeds `threshold`, AND
          (b) it is a local maximum within ±`min_distance` frames.

        This is a post-hoc inference utility and is not used during training.

        Args:
            density_map   (np.ndarray): shape (T,), values in [0, 1].
            threshold     (float):      minimum sigmoid value to consider.
            min_distance  (int):        minimum frames between detected peaks.

        Returns:
            Integer repetition count.
        """
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(
            density_map,
            height=threshold,
            distance=min_distance
        )
        return len(peaks)