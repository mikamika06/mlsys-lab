import warnings
import numpy as np


def run_oneshot(weights, recipe):
    if recipe.get("sparsity") == "2:4":
        warnings.warn(
            "Sparsity '2:4' is deprecated in oneshot API and will be removed in a future version. Use structured_sparsity='2:4' instead.",
            DeprecationWarning,
            stacklevel=1
        )
    quantized = {}
    for k, w in weights.items():
        scale = np.max(np.abs(w)) / 127.0 if np.max(np.abs(w)) > 0 else 1.0
        q = np.round(w / scale).astype(np.int8)
        quantized[k] = {"quantized_data": q, "scale": scale}
    return quantized
