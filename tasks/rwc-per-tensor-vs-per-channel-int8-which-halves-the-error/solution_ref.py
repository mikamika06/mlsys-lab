import numpy as np


def _sym_int8_quant(x: np.ndarray, axis) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if axis is None:
        amax = np.max(np.abs(x))
        scale = amax / 127.0 if amax > 0 else 1.0
    else:
        amax = np.max(np.abs(x), axis=axis, keepdims=True)
        scale = np.where(amax > 0, amax / 127.0, 1.0)
    q = np.clip(np.round(x / scale), -127, 127)
    return q * scale


def quant_granularity_errors(W: np.ndarray) -> dict:
    """Quantize W with symmetric INT8 both per-tensor and per-channel
    (per-row), report each reconstruction's MSE, and pick the winner
    (the scheme with the lower MSE)."""
    W = np.asarray(W, dtype=np.float64)

    W_tensor = _sym_int8_quant(W, axis=None)
    W_channel = _sym_int8_quant(W, axis=1)

    mse_tensor = float(np.mean((W - W_tensor) ** 2))
    mse_channel = float(np.mean((W - W_channel) ** 2))
    winner = "per_tensor" if mse_tensor < mse_channel else "per_channel"

    return {
        "mse_per_tensor": mse_tensor,
        "mse_per_channel": mse_channel,
        "winner": winner,
    }
