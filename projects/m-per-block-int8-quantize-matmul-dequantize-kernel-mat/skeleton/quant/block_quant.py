from typing import Tuple
import numpy as np


def quantize_per_tensor_int8(X: np.ndarray) -> Tuple[np.ndarray, float]:
    raise NotImplementedError


def quantize_per_block_int8(X: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError
