import numpy as np

def compute_divergences(set_a: list[list[int]], set_b: list[list[int]]) -> list[int]:
    raise NotImplementedError

def check_regression_gate(divergences: list[int], k: int, max_fail_fraction: float) -> bool:
    raise NotImplementedError

def analyze_near_ties(logits: np.ndarray, divergences: list[int], eps: float = 1e-5) -> float:
    raise NotImplementedError
