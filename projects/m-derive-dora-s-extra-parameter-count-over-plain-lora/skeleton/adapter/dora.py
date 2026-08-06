import numpy as np


def dora_extra_parameters(d_in: int, d_out: int, r: int) -> int:
    raise NotImplementedError


def dora_forward(w: np.ndarray, a: np.ndarray, b: np.ndarray, g: np.ndarray, alpha: float, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError
