import numpy as np


def round_e2m1(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def mxfp4(x: np.ndarray, block_size: int = 32) -> np.ndarray:
    raise NotImplementedError


def nvfp4(x: np.ndarray, block_size: int = 16, super_block: int = 256) -> np.ndarray:
    raise NotImplementedError


def effective_bits(fmt: str) -> float:
    raise NotImplementedError
