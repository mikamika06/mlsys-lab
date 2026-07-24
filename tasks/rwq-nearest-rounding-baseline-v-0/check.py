import numpy as np

from mlsys import scorers


def _ref_quantize_dequant(W, group_size):
    """Per-group asymmetric affine 4-bit RTN quantization (V=0 baseline)."""
    shape = W.shape
    flat = np.asarray(W, dtype=np.float64).ravel()
    n = flat.size
    codes = np.empty(n, dtype=np.uint8)
    dq = np.empty(n, dtype=np.float64)

    for start in range(0, n, group_size):
        g = flat[start:start + group_size]
        gmax = float(g.max())
        gmin = float(g.min())
        span = gmax - gmin
        scale = 1.0 if span == 0.0 else span / 15.0
        zero = float(np.clip(np.rint(-gmin / scale), 0, 15))
        code = np.clip(np.rint(g / scale) + zero, 0, 15).astype(np.uint8)
        end = start + len(g)
        codes[start:end] = code
        dq[start:end] = (code.astype(np.float64) - zero) * scale

    return codes.reshape(shape), dq.reshape(shape)


def _cases():
    rng = np.random.default_rng(0)
    cases = []
    cases.append((np.array([0.0, 1.5, -2.5, 7.0], dtype=np.float64), 4))
    cases.append((rng.normal(size=(5, 6)) * 2.0, 5))
    cases.append((rng.uniform(-3, 3, size=(4, 9)), 8))
    cases.append((np.full((3, 4), 2.0, dtype=np.float64), 6))  # constant group
    cases.append((rng.normal(loc=0.0, scale=0.02, size=(6, 11)), 8))  # size not / group_size
    return cases


def grade(sol, fx) -> dict:
    exact = 1.0
    worst = 0.0

    all_cases = list(_cases())
    all_cases.append((np.asarray(fx["ar_w"], dtype=np.float64), 32))

    for W, g in all_cases:
        ref_codes, ref_dq = _ref_quantize_dequant(W, g)
        try:
            got_codes, got_dq = sol.quantize_dequant_rtn_v0(np.array(W, copy=True), g)
        except Exception:
            return {"exact_match": 0.0, "max_abs_err": float("inf")}

        got_codes = np.asarray(got_codes)
        got_dq = np.asarray(got_dq, dtype=np.float64)

        if got_codes.shape != ref_codes.shape or got_dq.shape != ref_dq.shape:
            exact = 0.0
            worst = float("inf")
            continue

        codes_i = got_codes.astype(np.int64)
        if not np.array_equal(codes_i, ref_codes.astype(np.int64)):
            exact = 0.0
        if np.any(codes_i < 0) or np.any(codes_i > 15):
            exact = 0.0

        worst = max(worst, float(scorers.max_abs_err(ref_dq, got_dq)))

    return {"exact_match": exact, "max_abs_err": worst}
