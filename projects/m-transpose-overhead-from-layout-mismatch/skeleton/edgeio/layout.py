import numpy as np


def nhwc_to_nchw(arr: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def nchw_to_nhwc(arr: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def measure_transpose_bytes(shape: tuple, dtype: np.dtype) -> dict:
    raise NotImplementedError
