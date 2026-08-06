import numpy as np
from fp8kv.quant import quantize_e4m3_per_tensor, dequantize_e4m3_per_tensor


E5M2_MAX = 57344.0


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Mean Squared Error between two arrays."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def encode_e5m2(x: np.ndarray) -> np.ndarray:
    """Encode float32 array to uint8 containing bit-exact FP8 E5M2 representation."""
    f32 = np.asarray(x, dtype=np.float32)
    i32 = f32.view(np.uint32)

    sign = (i32 >> 31) & 0x1
    exp32 = (i32 >> 23) & 0xFF
    frac32 = i32 & 0x7FFFFF

    out = np.zeros(f32.shape, dtype=np.uint8)

    nan_mask = (exp32 == 255) & (frac32 != 0)
    out[nan_mask] = (sign[nan_mask] << 7) | 0x7F

    inf_mask = (exp32 == 255) & (frac32 == 0)
    out[inf_mask] = (sign[inf_mask] << 7) | 0x7C

    zero_mask = (exp32 == 0) & (frac32 == 0)
    out[zero_mask] = (sign[zero_mask] << 7)

    finite_mask = ~nan_mask & ~inf_mask & ~zero_mask
    val = np.abs(f32)

    too_large = finite_mask & (val > 57344.0)
    out[too_large] = (sign[too_large] << 7) | 0x7C

    valid_finite = finite_mask & ~too_large

    flat_val = val[valid_finite]
    flat_sign = sign[valid_finite]

    subnormal = flat_val < (2.0 ** -14)
    res = np.zeros(flat_val.shape, dtype=np.uint8)

    if np.any(subnormal):
        sub_vals = flat_val[subnormal]
        scaled = np.round(sub_vals * (2.0 ** 16))
        scaled = np.clip(scaled, 0, 3).astype(np.uint8)
        res[subnormal] = scaled

    normal = ~subnormal
    if np.any(normal):
        norm_vals = flat_val[normal]
        norm_i32 = norm_vals.view(np.uint32)
        e32 = ((norm_i32 >> 23) & 0xFF).astype(np.int32)
        f32_mant = norm_i32 & 0x7FFFFF

        e8 = e32 - 127 + 15

        mant_with_hidden = f32_mant | 0x800000
        m2_raw = mant_with_hidden >> 21
        rem = mant_with_hidden & 0x1FFFFF

        round_up = (rem > 0x100000) | ((rem == 0x100000) & ((m2_raw & 1) != 0))

        m2 = m2_raw & 0x3
        m2_final = np.where(round_up, m2 + 1, m2)
        e8_final = np.where(round_up & (m2_final == 4), e8 + 1, e8)
        m2_final = np.where(round_up & (m2_final == 4), 0, m2_final)

        overflow = (e8_final >= 31)

        encoded = np.where(
            overflow,
            31 << 2,
            (e8_final << 2) | m2_final
        )
        res[normal] = encoded.astype(np.uint8)

    out[valid_finite] = (flat_sign << 7) | res
    return out


def decode_e5m2(u: np.ndarray) -> np.ndarray:
    """Decode uint8 FP8 E5M2 representation to float32 array."""
    u8 = np.asarray(u, dtype=np.uint8)
    sign = ((u8 >> 7) & 0x1).astype(np.float32)
    s_mult = 1.0 - 2.0 * sign

    exp = (u8 >> 2) & 0x1F
    mant = u8 & 0x3

    out = np.zeros(u8.shape, dtype=np.float32)

    nan_mask = (exp == 31) & (mant != 0)
    out[nan_mask] = np.nan

    inf_mask = (exp == 31) & (mant == 0)
    out[inf_mask] = s_mult[inf_mask] * np.inf

    subnorm_mask = (exp == 0)
    out[subnorm_mask] = s_mult[subnorm_mask] * (2.0 ** -16) * mant[subnorm_mask]

    norm_mask = (exp > 0) & (exp < 31)
    out[norm_mask] = s_mult[norm_mask] * (2.0 ** (exp[norm_mask].astype(np.float32) - 15)) * (1.0 + mant[norm_mask].astype(np.float32) / 4.0)

    return out


def quantize_e5m2_per_tensor(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize array with per-tensor scale using absmax and max float value of E5M2 (57344.0)."""
    absmax = float(np.max(np.abs(x)))
    if absmax == 0.0:
        scale = 1.0
    else:
        scale = absmax / E5M2_MAX
    scaled_x = x / scale
    q = encode_e5m2(scaled_x)
    return q, scale


def dequantize_e5m2_per_tensor(u: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize uint8 FP8 E5M2 array given per-tensor scale factor."""
    decoded = decode_e5m2(u)
    return decoded * scale


def compare_formats_on_kv_dump(kv_dump: np.ndarray) -> dict[str, float]:
    """Return dictionary with 'e4m3_mse' and 'e5m2_mse' for given float32 KV cache dump."""
    q_e4m3, scale_e4m3 = quantize_e4m3_per_tensor(kv_dump)
    deq_e4m3 = dequantize_e4m3_per_tensor(q_e4m3, scale_e4m3)
    e4m3_mse = compute_mse(kv_dump, deq_e4m3)

    q_e5m2, scale_e5m2 = quantize_e5m2_per_tensor(kv_dump)
    deq_e5m2 = dequantize_e5m2_per_tensor(q_e5m2, scale_e5m2)
    e5m2_mse = compute_mse(kv_dump, deq_e5m2)

    return {
        "e4m3_mse": e4m3_mse,
        "e5m2_mse": e5m2_mse
    }
