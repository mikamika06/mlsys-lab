import numpy as np


def _e4m3_roundtrip(x: np.ndarray, scale: float) -> np.ndarray:
    """Per-tensor absmax-scaled E4M3 (4 exponent bits, 3 mantissa bits,
    max representable magnitude 448) quantize-then-dequantize."""
    y = np.asarray(x, dtype=np.float64) / scale
    y = np.clip(y, -448.0, 448.0)
    sign = np.sign(y)
    ay = np.abs(y)

    exp = np.floor(np.log2(np.maximum(ay, 2 ** -9)))
    exp = np.clip(exp, -6, 7)
    frac = ay / (2.0 ** exp) - 1.0
    mant = np.round(frac * 8.0) / 8.0
    val = (1.0 + mant) * (2.0 ** exp)
    val = np.where(ay < 2 ** -6, np.round(ay / (2 ** -9)) * (2 ** -9), val)
    val = np.where(ay == 0, 0.0, val)

    return sign * val * scale


def kv_fp8_reconstruction_mse(K: np.ndarray, V: np.ndarray) -> dict:
    """Quantize K and V to E4M3 with an independent PER-TENSOR absmax
    scale for each (scale = max(|X|) / 448), dequantize, and report each
    tensor's reconstruction MSE.

    K, V : arbitrary-shape float arrays.

    Returns {"mse_k": float, "mse_v": float}.
    """
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    sk = max(float(np.max(np.abs(K))) / 448.0, 1e-12)
    sv = max(float(np.max(np.abs(V))) / 448.0, 1e-12)

    K_hat = _e4m3_roundtrip(K, sk)
    V_hat = _e4m3_roundtrip(V, sv)

    mse_k = float(np.mean((K_hat - K) ** 2))
    mse_v = float(np.mean((V_hat - V) ** 2))
    return {"mse_k": mse_k, "mse_v": mse_v}
