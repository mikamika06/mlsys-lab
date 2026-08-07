import numpy as np


def compute_causal_mask(n_q: int, n_kv: int, alignment: str = "bottom_right") -> np.ndarray:
    """Generates a boolean causal mask (True = keep, False = mask out)."""
    raise NotImplementedError


def compute_offset(n_q: int, n_kv: int, alignment: str) -> int:
    """Computes the key index offset for query index 0."""
    raise NotImplementedError
