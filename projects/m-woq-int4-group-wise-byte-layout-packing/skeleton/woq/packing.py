import numpy as np


def pack_int4_groups(weights: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def unpack_int4_groups(packed: np.ndarray, scales: np.ndarray, group_size: int, original_shape: tuple[int, int]) -> np.ndarray:
    raise NotImplementedError
