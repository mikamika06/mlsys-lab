import numpy as np


class AWQModifier:
    def __init__(self, gamma: float = 0.5, n_grid: int = 10, min_clip: float = 0.4):
        raise NotImplementedError

    def inspect_and_scale(self, weights: np.ndarray, act_means: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def find_best_clipping(self, w_scaled: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
