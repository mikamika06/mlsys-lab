import numpy as np


def simulate_fp8_e4m3_quant(x: np.ndarray) -> np.ndarray:
    """Simulate FP8 E4M3 quantization (max abs val 448.0, 3 mantissa bits)."""
    max_val = 448.0
    abs_x = np.abs(x)
    scale = np.maximum(np.max(abs_x), 1e-8) / max_val
    scaled = x / scale

    sign = np.sign(scaled)
    abs_scaled = np.abs(scaled)
    clipped = np.clip(abs_scaled, 0.0, max_val)

    exponent = np.floor(np.log2(np.maximum(clipped, 1e-8)))
    exponent = np.clip(exponent, -6.0, 8.0)

    mantissa_step = 2.0 ** (exponent - 3.0)
    quantized_abs = np.round(clipped / mantissa_step) * mantissa_step

    res = sign * quantized_abs * scale
    return res


def derive_fp8_e4m3_error_bound(val_std: float, num_samples: int = 10000) -> float:
    """Derive expected RMS quantization error for normal distribution N(0, val_std^2)."""
    rng = np.random.default_rng(42)
    data = rng.normal(0.0, val_std, size=num_samples)
    quantized = simulate_fp8_e4m3_quant(data)
    rms_err = np.sqrt(np.mean((data - quantized) ** 2))
    return float(rms_err)
