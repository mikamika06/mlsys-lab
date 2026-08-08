"""Sensitivity measurement functions."""

import numpy as np


def quantize_weight(weight: np.ndarray, bits: int) -> np.ndarray:
    """Quantize weight array to specified bitwidth using min-max uniform quantization."""
    if bits >= 16:
        return weight.astype(np.float64)
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    w_min, w_max = weight.min(), weight.max()
    if w_min == w_max:
        return np.full_like(weight, w_min)
    scale = (w_max - w_min) / (qmax - qmin)
    zero_point = qmin - w_min / scale
    q = np.round(weight / scale + zero_point)
    q = np.clip(q, qmin, qmax)
    w_dequant = (q - zero_point) * scale
    return w_dequant


def measure_layer_sensitivity(model: dict, dataset: np.ndarray, candidate_bits: list[int]) -> dict:
    """
    Measure sensitivity per layer and candidate bitwidth.
    Returns dict: {layer_name: {bitwidth: float_sensitivity_score}}
    """
    sensitivity = {}
    base_out = _forward_unquantized(model, dataset)

    for name, w in model["layers"].items():
        sensitivity[name] = {}
        for b in candidate_bits:
            w_q = quantize_weight(w, b)
            temp_model = {"layers": dict(model["layers"])}
            temp_model["layers"][name] = w_q
            out_q = _forward_unquantized(temp_model, dataset)
            mse = float(np.mean((base_out - out_q) ** 2))
            sensitivity[name][b] = mse

    return sensitivity


def _forward_unquantized(model: dict, dataset: np.ndarray) -> np.ndarray:
    x = dataset
    for name, w in model["layers"].items():
        x = np.matmul(x, w)
        x = np.maximum(0, x)
    return x
