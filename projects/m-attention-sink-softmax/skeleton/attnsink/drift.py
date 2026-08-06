import numpy as np


def compute_rel_err(a: np.ndarray, ref: np.ndarray) -> float:
    raise NotImplementedError


def compute_drift(Q: np.ndarray, K: np.ndarray, V: np.ndarray, sink_size: int, window_size: int):
    raise NotImplementedError
