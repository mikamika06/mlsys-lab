import numpy as np


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Mean Squared Error between two arrays."""
    raise NotImplementedError


def encode_e5m2(x: np.ndarray) -> np.ndarray:
    """Encode float32 array to uint8 containing bit-exact FP8 E5M2 representation."""
    raise NotImplementedError


def decode_e5m2(u: np.ndarray) -> np.ndarray:
    """Decode uint8 FP8 E5M2 representation to float32 array."""
    raise NotImplementedError


def quantize_e5m2_per_tensor(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize array with per-tensor scale using absmax and max float value of E5M2 (57344.0)."""
    raise NotImplementedError


def dequantize_e5m2_per_tensor(u: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize uint8 FP8 E5M2 array given per-tensor scale factor."""
    raise NotImplementedError


def compare_formats_on_kv_dump(kv_dump: np.ndarray) -> dict[str, float]:
    """Return dictionary with 'e4m3_mse' and 'e5m2_mse' for given float32 KV cache dump."""
    raise NotImplementedError
