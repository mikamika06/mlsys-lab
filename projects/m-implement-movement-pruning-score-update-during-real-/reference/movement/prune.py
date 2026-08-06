import numpy as np


def update_movement_scores(scores: np.ndarray, weights: np.ndarray, grads: np.ndarray, lr: float) -> np.ndarray:
    return scores + weights * grads * lr


def reconstruct_trajectory(weight_series: list, grad_series: list, lr: float) -> np.ndarray:
    shape = weight_series[0].shape
    scores = np.zeros(shape, dtype=np.float32)
    for w, g in zip(weight_series, grad_series):
        scores = update_movement_scores(scores, w, g, lr)
    return scores


def compute_mask_overlap(movement_mask: np.ndarray, magnitude_mask: np.ndarray) -> float:
    intersection = np.logical_and(movement_mask, magnitude_mask).sum()
    union = np.logical_or(movement_mask, magnitude_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)
