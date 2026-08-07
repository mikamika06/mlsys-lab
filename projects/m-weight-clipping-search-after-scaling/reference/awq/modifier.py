import numpy as np
from awq.search import compute_activation_scales, search_weight_clipping


class AWQModifier:
    def __init__(self, gamma: float = 0.5, n_grid: int = 10, min_clip: float = 0.4):
        self.gamma = gamma
        self.n_grid = n_grid
        self.min_clip = min_clip

    def inspect_and_scale(self, weights: np.ndarray, act_means: np.ndarray) -> np.ndarray:
        scales = compute_activation_scales(act_means, self.gamma)
        return weights * scales[None, :]

    def find_best_clipping(self, w_scaled: np.ndarray, x: np.ndarray) -> np.ndarray:
        return search_weight_clipping(w_scaled, x, self.n_grid, self.min_clip)
