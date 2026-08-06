import numpy as np

def quantize_q8_0(x: np.ndarray) -> bytes:
    raise NotImplementedError

def dequantize_q8_0(data: bytes, shape: tuple) -> np.ndarray:
    raise NotImplementedError

def block_properties() -> dict:
    raise NotImplementedError

def compute_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    raise NotImplementedError

def rank_legacy_types(weights: np.ndarray) -> list:
    raise NotImplementedError
