import numpy as np


def update_movement_scores(scores: np.ndarray, weights: np.ndarray, grads: np.ndarray, lr: float) -> np.ndarray:
    raise NotImplementedError


def reconstruct_trajectory(weight_series: list, grad_series: list, lr: float) -> np.ndarray:
    raise NotImplementedError


def compute_mask_overlap(movement_mask: np.ndarray, magnitude_mask: np.ndarray) -> float:
    raise NotImplementedError
