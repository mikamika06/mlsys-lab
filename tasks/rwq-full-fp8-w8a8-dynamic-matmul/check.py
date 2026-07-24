import numpy as np
from mlsys import scorers

E4M3_MAX = 448.0


def _e4m3_grid_pos() -> np.ndarray:
    """All non-negative finite E4M3 (4 exponent bits, 3 mantissa bits, bias 7) values."""
    vals = set()
    for exp in range(16):
        for mant in range(8):
            if exp == 15 and mant == 7:
                continue  # NaN code point
            if exp == 0:
                v = (2.0 ** -6) * (mant / 8.0)  # subnormal
            else:
                v = (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)  # normal
            vals.add(v)
    return np.array(sorted(vals), dtype=np.float64)


_GRID = _e4m3_grid_pos()


def _cast_e4m3(x: np.ndarray) -> np.ndarray:
    """Round each element to the nearest representable E4M3 value (clamped to +-448)."""
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    absx = np.clip(np.abs(x), 0.0, E4M3_MAX)
    idx = np.searchsorted(_GRID, absx)
    idx = np.clip(idx, 1, len(_GRID) - 1)
    lo = _GRID[idx - 1]
    hi = _GRID[idx]
    snapped = np.where((hi - absx) < (absx - lo), hi, lo)
    return sign * snapped


def _oracle_fp8_dynamic_matmul(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    W64 = np.asarray(W, dtype=np.float64)
    X64 = np.asarray(X, dtype=np.float64)

    amax_w = float(np.max(np.abs(W64)))
    scale_w = amax_w / E4M3_MAX if amax_w > 0 else 1.0

    amax_x = np.max(np.abs(X64), axis=0)  # per-token (per-column of X)
    scale_x = np.where(amax_x > 0, amax_x / E4M3_MAX, 1.0)

    Wq = _cast_e4m3(W64 / scale_w)
    Xq = _cast_e4m3(X64 / scale_x[None, :])

    Y = (Wq @ Xq) * scale_w * scale_x[None, :]
    return Y


def grade(sol, fx) -> dict:
    """
    Builds a weight matrix W and an activation matrix X (with a couple of
    outlier entries, as real weights/activations have), computes:
      - the true full-precision matmul (fp64) as ground truth,
      - the FP8 E4M3 W8A8 dynamic-quantization oracle (per-tensor W scale,
        per-token X scale, cast both operands to the nearest representable
        E4M3 value, matmul, then dequantize),
    and grades the candidate on both closeness to the true matmul and
    closeness to the oracle's specific quantized result (so a solution that
    skips quantization entirely, or uses the wrong scale granularity, fails
    the second gate even if the first gate looks fine by luck).
    """
    rng = np.random.default_rng(0)
    M, K, N = 6, 8, 5
    W = (rng.normal(0, 1, size=(M, K)) * rng.uniform(0.1, 3.0)).astype(np.float32)
    X = rng.normal(0, 2, size=(K, N)).astype(np.float32)
    # inject a couple of outlier entries, as real weight/activation tensors have
    W[0, 0] *= 20.0
    X[:, 1] *= 15.0

    ref_fp32 = W.astype(np.float64) @ X.astype(np.float64)
    oracle = _oracle_fp8_dynamic_matmul(W, X)

    try:
        got = sol.fp8_dynamic_matmul(W.copy(), X.copy())
        got = np.asarray(got, dtype=np.float64)
    except Exception:
        return {"rel_err": float("inf"), "oracle_rel_err": float("inf")}

    if got.shape != ref_fp32.shape:
        return {"rel_err": float("inf"), "oracle_rel_err": float("inf")}

    return {
        "rel_err": scorers.rel_err(ref_fp32, got),
        "oracle_rel_err": scorers.rel_err(oracle, got),
    }
