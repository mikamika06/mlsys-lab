import numpy as np


def encode_e4m3(x: np.ndarray) -> np.ndarray:
    """Encode float32 array to uint8 containing bit-exact FP8 E4M3 representation."""
    raise NotImplementedError


def decode_e4m3(u: np.ndarray) -> np.ndarray:
    """Decode uint8 FP8 E4M3 representation to float32 array."""
    raise NotImplementedError


def quantize_e4m3_per_tensor(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize array with per-tensor scale using absmax and max float value of E4M3 (448.0)."""
    raise NotImplementedError


def dequantize_e4m3_per_tensor(u: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize uint8 FP8 E4M3 array given per-tensor scale factor."""
    raise NotImplementedError
