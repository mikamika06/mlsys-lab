import numpy as np


def compute_quant_params(
    float_min: float, float_max: float, qmin: int = 0, qmax: int = 255
) -> tuple[float, int]:
    """Compute scale and zero_point for quantizing float range into [qmin, qmax]."""
    raise NotImplementedError


def quantize_float_to_int8(
    x: np.ndarray, scale: float, zero_point: int, qmin: int = 0, qmax: int = 255
) -> np.ndarray:
    """Quantize floating point array to uint8/int8 array."""
    raise NotImplementedError


def match_input_scale_zp(
    mean: list[float],
    std: list[float],
    f_min: float = 0.0,
    f_max: float = 1.0,
    qmin: int = 0,
    qmax: int = 255,
) -> tuple[float, int]:
    """Compute overall effective scale and zero point for raw input given mean and std normalization."""
    raise NotImplementedError
