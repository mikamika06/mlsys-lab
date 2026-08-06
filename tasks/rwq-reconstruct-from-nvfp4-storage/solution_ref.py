import math
import numpy as np


def _decode_e4m3(codes):
    arr = np.asarray(codes, dtype=np.int64)
    shape = arr.shape
    flat = arr.ravel()
    out_list = []
    for code in flat:
        S = (code >> 7) & 1
        E = (code >> 3) & 0xF
        M = float(code & 0x7)
        sign = 1.0 if S == 0 else -1.0
        if E == 0:
            val = sign * math.ldexp(M / 8.0, -6)
        else:
            val = sign * math.ldexp(1.0 + M / 8.0, int(E - 7))
        out_list.append(val)
    return np.array(out_list, dtype=np.float64).reshape(shape)


def _decode_e2m1(codes):
    arr = np.asarray(codes, dtype=np.int64)
    shape = arr.shape
    flat = arr.ravel()
    out_list = []
    for code in flat:
        S = (code >> 3) & 1
        E = (code >> 1) & 0x3
        M = float(code & 0x1)
        sign = 1.0 if S == 0 else -1.0
        if E == 0:
            val = sign * (M / 2.0)
        else:
            val = sign * math.ldexp(1.0 + M / 2.0, int(E - 1))
        out_list.append(val)
    return np.array(out_list, dtype=np.float64).reshape(shape)


def nvfp4_reconstruct(global_scale, e4m3_block_codes: np.ndarray, e2m1_codes: np.ndarray) -> np.ndarray:
    g = float(global_scale)
    s_b = _decode_e4m3(e4m3_block_codes)
    q = _decode_e2m1(e2m1_codes)
    s_b_arr = np.asarray(s_b, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    n_blocks = s_b_arr.shape[0]
    n_cols = q_arr.shape[1]
    out_rows = []
    for i in range(n_blocks):
        sb_val = s_b_arr[i]
        row = []
        for j in range(n_cols):
            row.append(g * sb_val * q_arr[i, j])
        out_rows.append(row)
    return np.array(out_rows, dtype=np.float64)
