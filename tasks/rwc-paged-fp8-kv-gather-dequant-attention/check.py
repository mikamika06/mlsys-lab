import numpy as np

from mlsys import scorers


def _decode_bits(code: np.ndarray) -> np.ndarray:
    code = np.asarray(code, dtype=np.uint8)
    sign = np.where((code & 0x80) != 0, -1.0, 1.0)
    e = ((code >> 3) & 0x0F).astype(np.int64)
    m = (code & 0x07).astype(np.int64)
    normal = sign * (1.0 + m / 8.0) * np.exp2((e - 7).astype(np.float64))
    subnormal = sign * (m / 8.0) * np.exp2(-6.0)
    val = np.where(e == 0, subnormal, normal)
    val = np.where((e == 15) & (m == 7), np.nan, val)
    return val


_NONNEG_CODES = np.arange(0, 127, dtype=np.uint8)
_NONNEG_GRID = _decode_bits(_NONNEG_CODES)
_MAX_E4M3 = float(_NONNEG_GRID[-1])


def _encode_e4m3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sign_bit = np.where(np.signbit(x), np.uint8(0x80), np.uint8(0x00))
    av = np.clip(np.abs(x), 0.0, _MAX_E4M3)
    idx = np.clip(np.searchsorted(_NONNEG_GRID, av), 1, len(_NONNEG_GRID) - 1)
    lo_idx, hi_idx = idx - 1, idx
    lo, hi = _NONNEG_GRID[lo_idx], _NONNEG_GRID[hi_idx]
    d_lo, d_hi = av - lo, hi - av
    hi_even = (_NONNEG_CODES[hi_idx] & 1) == 0
    choose_hi = np.where(d_hi == d_lo, hi_even, d_hi < d_lo)
    mag_code = np.where(choose_hi, _NONNEG_CODES[hi_idx], _NONNEG_CODES[lo_idx]).astype(np.uint8)
    return (sign_bit | mag_code).astype(np.uint8)


def _oracle(k_codes_phys, v_codes_phys, k_scale, v_scale, block_table, seq_len, q):
    _, block_size, n_heads, D = k_codes_phys.shape
    L_b = block_table.shape[0]
    k_codes = k_codes_phys[block_table].reshape(L_b * block_size, n_heads, D)[:seq_len]
    v_codes = v_codes_phys[block_table].reshape(L_b * block_size, n_heads, D)[:seq_len]

    k_deq = _decode_bits(k_codes) * k_scale[None, :, None]
    v_deq = _decode_bits(v_codes) * v_scale[None, :, None]
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


def _build_case(rng, seq_len_true, n_heads, D, block_size, num_extra_phys=3, garbage_code=0x7E):
    L_b = int(np.ceil(seq_len_true / block_size))
    padded_len = L_b * block_size

    K = rng.normal(size=(seq_len_true, n_heads, D))
    V = rng.normal(size=(seq_len_true, n_heads, D))

    k_scale = np.zeros(n_heads)
    v_scale = np.zeros(n_heads)
    k_codes = np.zeros((seq_len_true, n_heads, D), dtype=np.uint8)
    v_codes = np.zeros((seq_len_true, n_heads, D), dtype=np.uint8)
    for h in range(n_heads):
        k_scale[h] = np.abs(K[:, h, :]).max() / _MAX_E4M3
        v_scale[h] = np.abs(V[:, h, :]).max() / _MAX_E4M3
        k_codes[:, h, :] = _encode_e4m3(K[:, h, :] / k_scale[h])
        v_codes[:, h, :] = _encode_e4m3(V[:, h, :] / v_scale[h])

    pad = padded_len - seq_len_true
    if pad > 0:
        garbage = np.full((pad, n_heads, D), garbage_code, dtype=np.uint8)
        k_codes = np.concatenate([k_codes, garbage], axis=0)
        v_codes = np.concatenate([v_codes, garbage], axis=0)

    num_phys = L_b + num_extra_phys
    block_table = rng.permutation(num_phys)[:L_b].astype(np.int64)

    k_phys = np.full((num_phys, block_size, n_heads, D), garbage_code, dtype=np.uint8)
    v_phys = np.full((num_phys, block_size, n_heads, D), garbage_code, dtype=np.uint8)
    for i in range(L_b):
        phys = int(block_table[i])
        k_phys[phys] = k_codes[i * block_size:(i + 1) * block_size]
        v_phys[phys] = v_codes[i * block_size:(i + 1) * block_size]

    q = rng.normal(size=(n_heads, D))
    return k_phys, v_phys, k_scale, v_scale, block_table, seq_len_true, q


def _cases():
    rng = np.random.default_rng(2026)
    specs = [
        (11, 4, 8, 4),    # last block missing 1 of 4
        (10, 3, 6, 3),    # last block missing 2 of 3
        (8, 2, 4, 4),     # exact multiple: no stale rows
        (25, 5, 6, 6),    # last block has only 1 valid row of 6
    ]
    out = []
    for seq_len_true, n_heads, D, block_size in specs:
        out.append(_build_case(rng, seq_len_true, n_heads, D, block_size))
    return out


def grade(sol, fx) -> dict:
    worst = 0.0
    for k_phys, v_phys, k_scale, v_scale, block_table, seq_len, q in _cases():
        expected = _oracle(k_phys, v_phys, k_scale, v_scale, block_table, seq_len, q)
        try:
            got = np.asarray(
                sol.paged_fp8_attention(
                    k_phys.copy(), v_phys.copy(), k_scale.copy(), v_scale.copy(),
                    block_table.copy(), seq_len, q.copy(),
                ),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != expected.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(scorers.max_abs_err(expected, got)))

    return {"max_abs_err": worst}
