import numpy as np


def _decode_e4m3(code: np.ndarray) -> np.ndarray:
    """Real E4M3FN bit-pattern decode: 1 sign, 4 exponent, 3 mantissa bits."""
    code = np.asarray(code, dtype=np.uint8)
    sign = np.where((code & 0x80) != 0, -1.0, 1.0)
    e = ((code >> 3) & 0x0F).astype(np.int64)
    m = (code & 0x07).astype(np.int64)
    normal = sign * (1.0 + m / 8.0) * np.exp2((e - 7).astype(np.float64))
    subnormal = sign * (m / 8.0) * np.exp2(-6.0)
    val = np.where(e == 0, subnormal, normal)
    val = np.where((e == 15) & (m == 7), np.nan, val)
    return val


def paged_fp8_attention(k_codes_phys, v_codes_phys, k_scale, v_scale, block_table, seq_len, q):
    """Gather paged FP8 (E4M3FN) KV via block_table, truncate to seq_len,
    dequantize per head, and run scaled dot-product attention."""
    _, block_size, n_heads, D = k_codes_phys.shape
    L_b = block_table.shape[0]

    k_codes = k_codes_phys[block_table].reshape(L_b * block_size, n_heads, D)[:seq_len]
    v_codes = v_codes_phys[block_table].reshape(L_b * block_size, n_heads, D)[:seq_len]

    k_scale = np.asarray(k_scale, dtype=np.float64)
    v_scale = np.asarray(v_scale, dtype=np.float64)
    k_deq = _decode_e4m3(k_codes) * k_scale[None, :, None]
    v_deq = _decode_e4m3(v_codes) * v_scale[None, :, None]

    q = np.asarray(q, dtype=np.float64)
    scale = 1.0 / np.sqrt(D)

    out = np.zeros((n_heads, D), dtype=np.float64)
    for h in range(n_heads):
        K = k_deq[:, h, :]
        V = v_deq[:, h, :]
        s = (K @ q[h]) * scale
        s = s - s.max()
        w = np.exp(s)
        w = w / w.sum()
        out[h] = w @ V

    return out
