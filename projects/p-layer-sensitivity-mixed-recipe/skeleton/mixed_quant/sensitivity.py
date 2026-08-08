"""Sensitivity measurement functions."""

import numpy as np


def quantize_weight(weight: np.ndarray, bits: int) -> np.ndarray:
    """Quantize weight array to specified bitwidth using min-max uniform quantization."""
    raise NotImplementedError


def measure_layer_sensitivity(model: dict, dataset: np.ndarray, candidate_bits: list[int]) -> dict:
    """
    Measure sensitivity per layer and candidate bitwidth.
    Returns dict: {layer_name: {bitwidth: float_sensitivity_score}}
    """
    raise NotImplementedError
