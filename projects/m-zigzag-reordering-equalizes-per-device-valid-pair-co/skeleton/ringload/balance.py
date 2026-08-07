import numpy as np


def compute_per_device_pairs(seq_len: int, chunk_size: int, world_size: int, scheme: str) -> np.ndarray:
    raise NotImplementedError


def compute_imbalance_ratio(per_device_pairs: np.ndarray) -> float:
    raise NotImplementedError


def compare_balancing_schemes(seq_len: int, chunk_size: int, world_size: int) -> dict:
    raise NotImplementedError
