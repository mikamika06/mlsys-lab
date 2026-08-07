import numpy as np


def compute_activation_scales(act_means: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    raise NotImplementedError


def quantize_asym_int4(w: np.ndarray, clip_ratio: float) -> np.ndarray:
    raise NotImplementedError


def search_weight_clipping(w_scaled: np.ndarray, x: np.ndarray, n_grid: int = 10, min_clip: float = 0.4) -> np.ndarray:
    raise NotImplementedError
