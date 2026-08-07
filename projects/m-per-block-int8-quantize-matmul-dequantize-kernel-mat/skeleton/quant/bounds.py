import numpy as np


def derive_fp8_e4m3_error_bound(val_std: float, num_samples: int = 10000) -> float:
    """Derive the expected RMS quantization error bound for FP8 E4M3 on a normal distribution."""
    raise NotImplementedError


def simulate_fp8_e4m3_quant(x: np.ndarray) -> np.ndarray:
    """Simulate FP8 E4M3 quantization and dequantization on a numpy array."""
    raise NotImplementedError
