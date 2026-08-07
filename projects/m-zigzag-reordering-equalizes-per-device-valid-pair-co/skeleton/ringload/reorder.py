import numpy as np


def compute_valid_pairs_causal(q_indices: np.ndarray, k_indices: np.ndarray) -> int:
    raise NotImplementedError


def assign_chunks_naive(num_chunks: int, world_size: int) -> np.ndarray:
    raise NotImplementedError


def assign_chunks_striped(num_chunks: int, world_size: int) -> np.ndarray:
    raise NotImplementedError


def assign_chunks_zigzag(num_chunks: int, world_size: int) -> np.ndarray:
    raise NotImplementedError
