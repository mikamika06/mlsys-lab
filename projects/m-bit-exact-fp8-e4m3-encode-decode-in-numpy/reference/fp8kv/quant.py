import numpy as np


E4M3_MAX = 448.0


def encode_e4m3(x: np.ndarray) -> np.ndarray:
    """Encode float32 array to uint8 containing bit-exact FP8 E4M3 representation."""
    f32 = np.asarray(x, dtype=np.float32)
    i32 = f32.view(np.uint32)

    sign = (i32 >> 31) & 0x1
    exp32 = (i32 >> 23) & 0xFF
    frac32 = i32 & 0x7FFFFF

    out = np.zeros(f32.shape, dtype=np.uint8)

    nan_mask = (exp32 == 255) & (frac32 != 0)
    out[nan_mask] = (sign[nan_mask] << 7) | 0x7F

    inf_mask = (exp32 == 255) & (frac32 == 0)
    out[inf_mask] = (sign[inf_mask] << 7) | 0x7F

    zero_mask = (exp32 == 0) & (frac32 == 0)
    out[zero_mask] = (sign[zero_mask] << 7)

    finite_mask = ~nan_mask & ~inf_mask & ~zero_mask

    val = np.abs(f32)

    too_large = finite_mask & (val > 448.0)
    out[too_large] = (sign[too_large] << 7) | 0x7E

    valid_finite = finite_mask & ~too_large

    flat_val = val[valid_finite]
    flat_sign = sign[valid_finite]

    subnormal = flat_val < (2.0 ** -6)

    res = np.zeros(flat_val.shape, dtype=np.uint8)

    if np.any(subnormal):
        sub_vals = flat_val[subnormal]
        scaled = np.round(sub_vals * (2.0 ** 9))
        scaled = np.clip(scaled, 0, 7).astype(np.uint8)
        res[subnormal] = scaled

    normal = ~subnormal
    if np.any(normal):
        norm_vals = flat_val[normal]
        norm_i32 = norm_vals.view(np.uint32)
        e32 = ((norm_i32 >> 23) & 0xFF).astype(np.int32)
        f32_mant = norm_i32 & 0x7FFFFF

        e8 = e32 - 127 + 7

        mant_with_hidden = f32_mant | 0x800000

        m3_raw = mant_with_hidden >> 20
        rem = mant_with_hidden & 0xFFFFF

        round_up = (rem > 0x80000) | ((rem == 0x80000) & ((m3_raw & 1) != 0))

        m3 = m3_raw & 0x7
        m3_final = np.where(round_up, m3 + 1, m3)
        e8_final = np.where(round_up & (m3_final == 8), e8 + 1, e8)
        m3_final = np.where(round_up & (m3_final == 8), 0, m3_final)

        overflow = (e8_final == 15) & (m3_final > 6)

        encoded = np.where(
            overflow,
            (e8_final << 3) | 6,
            (e8_final << 3) | m3_final
        )
        res[normal] = encoded.astype(np.uint8)

    out[valid_finite] = (flat_sign << 7) | res
    return out


def decode_e4m3(u: np.ndarray) -> np.ndarray:
    """Decode uint8 FP8 E4M3 representation to float32 array."""
    u8 = np.asarray(u, dtype=np.uint8)
    sign = ((u8 >> 7) & 0x1).astype(np.float32)
    s_mult = 1.0 - 2.0 * sign

    exp = (u8 >> 3) & 0xF
    mant = u8 & 0x7

    out = np.zeros(u8.shape, dtype=np.float32)

    nan_mask = (exp == 15) & (mant == 7)
    out[nan_mask] = np.nan

    subnorm_mask = (exp == 0)
    out[subnorm_mask] = s_mult[subnorm_mask] * (2.0 ** -9) * mant[subnorm_mask]

    norm_mask = (exp > 0) & ~nan_mask
    out[norm_mask] = s_mult[norm_mask] * (2.0 ** (exp[norm_mask].astype(np.float32) - 7)) * (1.0 + mant[norm_mask].astype(np.float32) / 8.0)

    return out


def quantize_e4m3_per_tensor(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize array with per-tensor scale using absmax and max float value of E4M3 (448.0)."""
    absmax = float(np.max(np.abs(x)))
    if absmax == 0.0:
        scale = 1.0
    else:
        scale = absmax / E4M3_MAX
    scaled_x = x / scale
    q = encode_e4m3(scaled_x)
    return q, scale


def dequantize_e4m3_per_tensor(u: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize uint8 FP8 E4M3 array given per-tensor scale factor."""
    decoded = decode_e4m3(u)
    return decoded * scale
