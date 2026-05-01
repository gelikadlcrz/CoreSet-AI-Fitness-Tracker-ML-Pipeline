import numpy as np
import torch
import pandas as pd

class CoreSetEvaluator:
    def __init__(self, exercise_classes=['squat', 'push_up', 'bench_press', 'pull_up']):
        self.classes = exercise_classes
        self.num_classes = len(exercise_classes)

    def calculate_normalized_mae(self, predictions, ground_truths):
        results = pd.DataFrame({'pred': predictions, 'gt': ground_truths})
        mask = results['gt'] > 0
        mae_raw = np.abs(results['pred'] - results['gt'])
        results['normalized_ae'] = 0.0
        results.loc[mask, 'normalized_ae'] = mae_raw[mask] / results.loc[mask, 'gt']
        return results['normalized_ae'].mean()

    def calculate_rmse(self, predictions, ground_truths):
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        mse = np.mean((predictions - ground_truths)**2)
        return np.sqrt(mse)

    def calculate_obo_accuracy(self, predictions, ground_truths, tolerance=1):
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        abs_diff = np.abs(predictions - ground_truths)
        correct_obo = (abs_diff <= tolerance).sum()
        return correct_obo / len(predictions)

    def calculate_classification_accuracy(self, logits, ground_truths):
        probabilities = torch.softmax(logits, dim=1)
        _, top_class = probabilities.topk(1, dim=1)
        top_class = top_class.squeeze()
        equals = (top_class == ground_truths).sum()
        return equals.item() / len(ground_truths)