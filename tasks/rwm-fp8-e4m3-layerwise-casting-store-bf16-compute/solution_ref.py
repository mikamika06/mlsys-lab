import math
import numpy as np


def _e4m3_table():
    codes = np.arange(256, dtype=np.uint8)
    val = np.empty(256, dtype=np.float64)
    for i in range(256):
        c = codes[i]
        sign = float((c >> 7) & 1)
        exp = float((c >> 3) & 0xF)
        mant = float(c & 0x7)
        s = 1.0 - 2.0 * sign
        subnormal = (exp == 0.0)
        if subnormal:
            v = s * (2.0 ** (-6.0)) * (mant / 8.0)
        else:
            v = s * (2.0 ** (exp - 7.0)) * (1.0 + mant / 8.0)
        if exp == 15.0 and mant == 7.0:
            val[i] = float('nan')
        else:
            val[i] = v
    return val


_TABLE = _e4m3_table()
_FINITE_MASK = ~np.isnan(_TABLE)
_FINITE_VALS = _TABLE[_FINITE_MASK]
_FINITE_CODES = np.arange(256, dtype=np.uint8)[_FINITE_MASK]


def _e4m3_encode(x):
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    out_codes = np.empty(flat.shape, dtype=np.uint8)
    for i in range(flat.shape[0]):
        val = flat[i]
        best_idx = 0
        min_diff = float('inf')
        for j in range(_FINITE_VALS.shape[0]):
            fv = _FINITE_VALS[j]
            diff = val - fv
            if diff < 0.0:
                diff = -diff
            if diff < min_diff:
                min_diff = diff
                best_idx = j
        out_codes[i] = _FINITE_CODES[best_idx]
    return out_codes.reshape(x.shape)


def _e4m3_decode(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    flat_codes = codes.ravel()
    out = np.empty(flat_codes.shape, dtype=np.float64)
    for i in range(flat_codes.shape[0]):
        out[i] = _TABLE[flat_codes[i]]
    return out.reshape(codes.shape)


def _to_bf16(x):
    x32 = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    bits = x32.view(np.uint32)
    flat_bits = bits.ravel()
    out_bits = np.empty(flat_bits.shape, dtype=np.uint32)
    for i in range(flat_bits.shape[0]):
        b = flat_bits[i]
        rounding_bias = ((b >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
        out_bits[i] = (b + rounding_bias) & np.uint32(0xFFFF0000)
    bits_rounded = out_bits.reshape(x32.shape)
    return bits_rounded.view(np.float32).reshape(x32.shape)


def cast_and_matmul_fp8e4m3(W: np.ndarray, X: np.ndarray):
    """
    Store W in E4M3 (nearest-value 8-bit codes), upcast the decoded grid
    value to bfloat16 precision for compute, snap X to bfloat16 too, and
    matmul with float32 accumulation. Returns (Y, codes).
    """
    codes = _e4m3_encode(W)
    w_fp8 = _e4m3_decode(codes)
    w_bf16 = _to_bf16(w_fp8)
    x_bf16 = _to_bf16(X)
    
    x_mat = x_bf16.astype(np.float32)
    w_mat = w_bf16.astype(np.float32)
    
    m = x_mat.shape[0]
    k = x_mat.shape[1]
    n = w_mat.shape[1]
    
    Y = np.empty((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += float(x_mat[i, p]) * float(w_mat[p, j])
            Y[i, j] = acc
            
    return Y, codes
