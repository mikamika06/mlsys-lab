import numpy as np


def compute_quant_params(
    float_min: float, float_max: float, qmin: int = 0, qmax: int = 255
) -> tuple[float, int]:
    """Compute scale and zero_point for quantizing float range into [qmin, qmax]."""
    scale = (float_max - float_min) / (qmax - qmin)
    if scale == 0:
        scale = 1.0
    zero_point = int(round(qmin - float_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    return float(scale), int(zero_point)


def quantize_float_to_int8(
    x: np.ndarray, scale: float, zero_point: int, qmin: int = 0, qmax: int = 255
) -> np.ndarray:
    """Quantize floating point array to uint8/int8 array."""
    q = np.round(x / scale) + zero_point
    return np.clip(q, qmin, qmax).astype(np.uint8)


def match_input_scale_zp(
    mean: list[float],
    std: list[float],
    f_min: float = 0.0,
    f_max: float = 1.0,
    qmin: int = 0,
    qmax: int = 255,
) -> tuple[float, int]:
    """Compute overall effective scale and zero point for raw input given mean and std normalization."""
    avg_mean = float(np.mean(mean))
    avg_std = float(np.mean(std))
    norm_min = (f_min - avg_mean) / avg_std
    norm_max = (f_max - avg_mean) / avg_std
    return compute_quant_params(norm_min, norm_max, qmin, qmax)
