import numpy as np


def per_token_ce(logits16: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-token cross-entropy for float16 logits `logits16` (N, V) and int targets (N,).

    Returns a float32 array of shape (N,).
    """
    raise NotImplementedError('your code here')


def mean_ce_fp32(logits16: np.ndarray, targets: np.ndarray) -> float:
    """Mean of `per_token_ce`, accumulated in float32 (or wider). Returns a Python float."""
    raise NotImplementedError('your code here')


def fp16_accum_stall_index(losses: np.ndarray) -> int:
    """Index of the first loss absorbed by a sequential float16 accumulator, else -1."""
    raise NotImplementedError('your code here')
