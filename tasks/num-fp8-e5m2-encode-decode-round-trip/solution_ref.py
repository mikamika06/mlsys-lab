import numpy as np


def _decode_e5m2_grid(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    s = (codes >> 7) & 1
    e = ((codes >> 2) & 0x1F).astype(np.int64)
    m = (codes & 0x3).astype(np.int64)
    sign = np.where(s == 1, -1.0, 1.0)
    normal_val = sign * np.exp2((e - 15).astype(np.float64)) * (1.0 + m / 4.0)
    sub_val = sign * np.exp2(-14.0) * (m / 4.0)
    return np.where(e == 0, sub_val, normal_val)


_MAG_GRID = _decode_e5m2_grid(np.arange(124, dtype=np.uint8))
_NEXT_VIRTUAL = 65536.0
_EXT_GRID = np.concatenate([_MAG_GRID, [_NEXT_VIRTUAL]])


def encode_e5m2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).astype(np.float64)
    shape = values.shape
    flat = values.ravel()
    sign_bit = np.signbit(flat).astype(np.uint8)
    av = np.abs(flat)

    out = np.zeros(flat.shape, dtype=np.uint8)

    nan_mask = np.isnan(flat)
    inf_mask = np.isinf(flat)
    finite_mask = ~(nan_mask | inf_mask)

    if np.any(finite_mask):
        av_f = av[finite_mask]
        av_c = np.minimum(av_f, _NEXT_VIRTUAL)
        idx = np.searchsorted(_EXT_GRID, av_c, side="left")
        idx = np.clip(idx, 1, len(_EXT_GRID) - 1)
        lo = idx - 1
        hi = idx
        d_lo = av_c - _EXT_GRID[lo]
        d_hi = _EXT_GRID[hi] - av_c
        pick_hi = d_hi < d_lo
        tie = d_hi == d_lo
        hi_even = (hi % 2) == 0
        pick_hi = np.where(tie, hi_even, pick_hi)
        chosen = np.where(pick_hi, hi, lo)
        chosen = np.where(av_f > _NEXT_VIRTUAL, len(_EXT_GRID) - 1, chosen)
        is_inf_result = chosen >= (len(_EXT_GRID) - 1)
        code_finite = np.where(is_inf_result, 0x7C, chosen).astype(np.uint8)
        out[finite_mask] = code_finite

    out[inf_mask] = 0x7C
    out[nan_mask] = 0x7F

    out = (sign_bit << 7) | out
    return out.reshape(shape)
