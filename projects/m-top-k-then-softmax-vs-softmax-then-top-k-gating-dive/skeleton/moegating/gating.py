import numpy as np


def top_k_then_softmax(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def softmax_then_top_k(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def analyze_gating_divergence(
    logits: np.ndarray, k: int
) -> dict[str, float | np.ndarray]:
    raise NotImplementedError
