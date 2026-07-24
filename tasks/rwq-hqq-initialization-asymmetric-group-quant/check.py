import numpy as np

from mlsys import scorers


def _oracle(W, group_size, nbits):
    """Independent re-implementation of the HQQ asymmetric group-quant init."""
    shape = W.shape
    flat = np.asarray(W, dtype=np.float64).ravel()
    n = flat.size
    qmax = 2 ** nbits - 1

    codes = np.empty(n, dtype=np.uint8)
    dequant = np.empty(n, dtype=np.float64)
    scales = []
    zeros = []

    for start in range(0, n, group_size):
        g = flat[start:start + group_size]
        gmax = float(g.max())
        gmin = float(g.min())
        span = gmax - gmin
        scale = 1.0 if span == 0.0 else span / qmax
        zero = float(np.round(-gmin / scale))
        code = np.clip(np.round(g / scale) + zero, 0, qmax).astype(np.uint8)

        codes[start:start + len(g)] = code
        dequant[start:start + len(g)] = (code.astype(np.float64) - zero) * scale
        scales.append(scale)
        zeros.append(zero)

    return (
        codes.reshape(shape),
        np.asarray(scales, dtype=np.float64),
        np.asarray(zeros, dtype=np.float64),
        dequant.reshape(shape),
    )


def _cases():
    rng = np.random.default_rng(11)
    cases = []
    cases.append((np.array([0.0, 1.5, -2.5, 7.0, 3.0, -1.0], dtype=np.float64), 2, 4))
    cases.append((rng.normal(size=(8, 16)) * 3.0, 64, 4))
    cases.append((rng.uniform(-5, 5, size=(6, 7)), 8, 3))
    cases.append((np.zeros((2, 6), dtype=np.float64), 4, 4))
    cases.append((rng.normal(loc=10.0, scale=2.0, size=(3, 17)), 8, 4))
    cases.append((np.full((2, 3), 5.0, dtype=np.float64), 4, 8))
    cases.append((rng.normal(size=(4, 65)).astype(np.float32), 64, 4))
    return cases


def grade(sol, fx) -> dict:
    codes_exact = 1.0
    worst_abs = 0.0

    for W, group_size, nbits in _cases():
        ref_codes, ref_scale, ref_zero, ref_dequant = _oracle(W, group_size, nbits)
        try:
            got = sol.hqq_init(np.array(W, copy=True), group_size, nbits)
            got_codes, got_scale, got_zero, got_dequant = got
        except Exception:
            return {"codes_exact": 0.0, "max_abs_err": float("inf")}

        got_codes = np.asarray(got_codes)
        got_dequant = np.asarray(got_dequant, dtype=np.float64)

        if got_codes.shape != ref_codes.shape or got_dequant.shape != ref_dequant.shape:
            return {"codes_exact": 0.0, "max_abs_err": float("inf")}

        qmax = 2 ** nbits - 1
        codes_int = got_codes.astype(np.int64)
        if np.any(codes_int < 0) or np.any(codes_int > qmax):
            codes_exact = 0.0
        if not np.array_equal(codes_int, ref_codes.astype(np.int64)):
            codes_exact = 0.0

        if not np.all(np.isfinite(got_dequant)):
            return {"codes_exact": codes_exact, "max_abs_err": float("inf")}

        worst_abs = max(worst_abs, float(scorers.max_abs_err(ref_dequant, got_dequant)))

    return {"codes_exact": codes_exact, "max_abs_err": worst_abs}
