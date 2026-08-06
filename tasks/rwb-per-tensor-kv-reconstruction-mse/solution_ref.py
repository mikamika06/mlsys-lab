import math
import numpy as np


def _e4m3_roundtrip(x: np.ndarray, scale: float) -> np.ndarray:
    """Per-tensor absmax-scaled E4M3 (4 exponent bits, 3 mantissa bits,
    max representable magnitude 448) quantize-then-dequantize."""
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty(x_arr.shape, dtype=np.float64)
    flat_x = x_arr.reshape(-1)
    flat_out = out.reshape(-1)

    for i in range(flat_x.size):
        val_x = float(flat_x[i])
        y = val_x / scale
        if y > 448.0:
            y = 448.0
        elif y < -448.0:
            y = -448.0

        if y > 0.0:
            sign = 1.0
        elif y < 0.0:
            sign = -1.0
        else:
            sign = 0.0

        ay = abs(y)

        max_ay = ay if ay > 0.001953125 else 0.001953125
        exp = math.floor(math.log2(max_ay))
        if exp < -6:
            exp = -6
        elif exp > 7:
            exp = 7

        pow2_exp = math.pow(2.0, exp)
        frac = ay / pow2_exp - 1.0
        mant = round(frac * 8.0) / 8.0
        val = (1.0 + mant) * pow2_exp

        if ay < 0.015625:
            val = round(ay / 0.001953125) * 0.001953125

        if ay == 0.0:
            val = 0.0

        flat_out[i] = sign * val * scale

    return out


def kv_fp8_reconstruction_mse(K: np.ndarray, V: np.ndarray) -> dict:
    """Quantize K and V to E4M3 with an independent PER-TENSOR absmax
    scale for each (scale = max(|X|) / 448), dequantize, and report each
    tensor's reconstruction MSE.

    K, V : arbitrary-shape float arrays.

    Returns {"mse_k": float, "mse_v": float}.
    """
    K_arr = np.asarray(K, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)

    flat_k = K_arr.reshape(-1)
    max_k = 0.0
    for i in range(flat_k.size):
        val = abs(float(flat_k[i]))
        if val > max_k:
            max_k = val

    sk = max_k / 448.0
    if sk < 1e-12:
        sk = 1e-12

    flat_v = V_arr.reshape(-1)
    max_v = 0.0
    for i in range(flat_v.size):
        val = abs(float(flat_v[i]))
        if val > max_v:
            max_v = val

    sv = max_v / 448.0
    if sv < 1e-12:
        sv = 1e-12

    K_hat = _e4m3_roundtrip(K_arr, sk)
    V_hat = _e4m3_roundtrip(V_arr, sv)

    flat_k_hat = K_hat.reshape(-1)
    sum_sq_k = 0.0
    for i in range(flat_k.size):
        diff = float(flat_k_hat[i]) - float(flat_k[i])
        sum_sq_k += diff * diff
    mse_k = float(sum_sq_k / flat_k.size)

    flat_v_hat = V_hat.reshape(-1)
    sum_sq_v = 0.0
    for i in range(flat_v.size):
        diff = float(flat_v_hat[i]) - float(flat_v[i])
        sum_sq_v += diff * diff
    mse_v = float(sum_sq_v / flat_v.size)

    return {"mse_k": mse_k, "mse_v": mse_v}
