import numpy as np
from e4m3.quant import decode_e4m3, encode_e4m3


def derive_fp8_scale(arr: np.ndarray, max_fp8: float = 448.0) -> float:
    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0.0:
        return 1.0
    return max_fp8 / max_abs


def _quant_decode_e5m2(arr: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float32).ravel()
    out = np.zeros_like(flat)
    max_val = 57344.0

    for i, x in enumerate(flat):
        if np.isnan(x):
            out[i] = np.nan
            continue
        sign = -1.0 if np.copysign(1.0, x) < 0 else 1.0
        abs_x = abs(float(x))

        if abs_x >= max_val:
            out[i] = sign * max_val
            continue
        if abs_x < (2.0 ** (-16)) / 2.0:
            out[i] = sign * 0.0
            continue
        if abs_x < 2.0 ** (-14):
            frac = abs_x / (2.0 ** (-16))
            mant = int(round(frac))
            out[i] = sign * (2.0 ** (-14)) * (mant / 4.0)
            continue

        exp = int(np.floor(np.log2(abs_x)))
        mant = int(round(((abs_x / (2.0 ** exp)) - 1.0) * 4.0))
        if mant == 4:
            mant = 0
            exp += 1
        out[i] = sign * (2.0 ** exp) * (1.0 + mant / 4.0)

    return out.reshape(arr.shape)


def _quant_decode_int8(arr: np.ndarray, max_fp8: float = 448.0) -> np.ndarray:
    scaled = arr * (127.0 / max_fp8)
    clipped = np.clip(np.round(scaled), -128, 127)
    return clipped * (max_fp8 / 127.0)


def compare_formats(arr: np.ndarray, scale: float) -> dict:
    scaled = arr * scale

    e4m3_enc = encode_e4m3(scaled)
    e4m3_dec = decode_e4m3(e4m3_enc) / scale
    e4m3_mse = float(np.mean((arr - e4m3_dec) ** 2))

    e5m2_dec = _quant_decode_e5m2(scaled) / scale
    e5m2_mse = float(np.mean((arr - e5m2_dec) ** 2))

    int8_dec = _quant_decode_int8(scaled, max_fp8=448.0) / scale
    int8_mse = float(np.mean((arr - int8_dec) ** 2))

    return {
        "e4m3_mse": e4m3_mse,
        "e5m2_mse": e5m2_mse,
        "int8_mse": int8_mse,
    }
