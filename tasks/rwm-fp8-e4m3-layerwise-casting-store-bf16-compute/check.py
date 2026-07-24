import numpy as np

from mlsys import scorers


def _e4m3_table():
    codes = np.arange(256, dtype=np.uint8)
    sign = ((codes >> 7) & 1).astype(np.float64)
    exp = ((codes >> 3) & 0xF).astype(np.float64)
    mant = (codes & 0x7).astype(np.float64)
    s = 1.0 - 2.0 * sign
    subnormal = exp == 0
    normal_val = s * (2.0 ** (exp - 7.0)) * (1.0 + mant / 8.0)
    subnormal_val = s * (2.0 ** (-6.0)) * (mant / 8.0)
    val = np.where(subnormal, subnormal_val, normal_val)
    nan_mask = (exp == 15) & (mant == 7)
    val = np.where(nan_mask, np.nan, val)
    return val


_TABLE = _e4m3_table()
_FINITE_MASK = ~np.isnan(_TABLE)
_FINITE_VALS = _TABLE[_FINITE_MASK]
_FINITE_CODES = np.arange(256, dtype=np.uint8)[_FINITE_MASK]


def _e4m3_encode(x):
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    diff = np.abs(flat[:, None] - _FINITE_VALS[None, :])
    idx = np.argmin(diff, axis=1)
    return _FINITE_CODES[idx].reshape(x.shape).astype(np.uint8)


def _e4m3_decode(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    return _TABLE[codes]


def _to_bf16(x):
    x32 = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    bits = x32.view(np.uint32)
    rounding_bias = ((bits >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    bits_rounded = (bits + rounding_bias) & np.uint32(0xFFFF0000)
    return bits_rounded.view(np.float32).reshape(x32.shape)


def _oracle(W, X):
    codes = _e4m3_encode(W)
    w_fp8 = _e4m3_decode(codes)
    w_bf16 = _to_bf16(w_fp8)
    x_bf16 = _to_bf16(X)
    Y = x_bf16.astype(np.float32) @ w_bf16.astype(np.float32)
    return Y, codes


def grade(sol, fx) -> dict:
    """
    The grader builds several random (W, X) pairs (X pre-snapped to the
    bfloat16 grid), computes the E4M3-storage + bfloat16-compute matmul with
    a NumPy oracle, and compares against the submission's Y (rel_err) and
    the byte size of the returned storage codes vs W's fp32 size (size_ratio).
    """
    rng = np.random.default_rng(0)
    worst_rel = 0.0
    worst_ratio = float("inf")
    for _ in range(6):
        try:
            k = int(rng.integers(2, 12))
            n = int(rng.integers(2, 10))
            m = int(rng.integers(2, 10))
            scale = float(rng.uniform(0.1, 100.0))
            W = (rng.normal(size=(k, n)) * scale).astype(np.float32)
            X_raw = (rng.normal(size=(m, k)) * scale).astype(np.float32)
            X = _to_bf16(X_raw)

            Y_exp, codes_exp = _oracle(W, X)
            got = sol.cast_and_matmul_fp8e4m3(W.copy(), X.copy())
            Y_got, codes_got = got

            rel = scorers.rel_err(Y_exp, np.asarray(Y_got))
            ratio = scorers.size_ratio(W, np.asarray(codes_got))
        except Exception:
            rel = float("inf")
            ratio = 0.0

        worst_rel = max(worst_rel, rel)
        worst_ratio = min(worst_ratio, ratio)

    return {"rel_err": worst_rel, "size_ratio": worst_ratio}
