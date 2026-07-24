import numpy as np

from mlsys import scorers


def _ref_quantize(W, group_size):
    """GPTQ-style per-group asymmetric affine 4-bit quant over the raveled array."""
    flat = np.asarray(W, dtype=np.float64).ravel()
    n = flat.size
    codes = np.empty(n, dtype=np.uint8)
    scales = []
    zeros = []
    for start in range(0, n, group_size):
        g = flat[start:start + group_size]
        gmax = float(g.max())
        gmin = float(g.min())
        span = gmax - gmin
        scale = 1.0 if span == 0.0 else span / 15.0
        zero = float(np.clip(np.rint(-gmin / scale), 0, 15))
        code = np.clip(np.rint(g / scale) + zero, 0, 15).astype(np.uint8)
        codes[start:start + len(g)] = code
        scales.append(scale)
        zeros.append(zero)
    return codes.reshape(W.shape), np.asarray(scales, dtype=np.float64), np.asarray(zeros, dtype=np.float64)


def _ref_dequant(codes, scale, zero, group_size, shape):
    flat = np.asarray(codes, dtype=np.float64).ravel()
    n = flat.size
    out = np.empty(n, dtype=np.float64)
    for i in range(len(scale)):
        start = i * group_size
        end = min(n, start + group_size)
        out[start:end] = (flat[start:end] - zero[i]) * scale[i]
    return out.reshape(shape)


def _cases():
    rng = np.random.default_rng(0)
    cases = []
    cases.append((np.array([0.0, 1.5, -2.5, 7.0, 3.0, -1.0], dtype=np.float64), 2))
    cases.append((rng.normal(size=(4, 5)) * 3.0, 3))
    cases.append((rng.uniform(-5, 5, size=(6, 7)), 8))
    cases.append((np.zeros((2, 6), dtype=np.float64), 4))
    cases.append((rng.normal(loc=10.0, scale=2.0, size=(3, 17)), 8))
    cases.append((np.full((2, 3), 5.0, dtype=np.float64), 4))
    return cases


def grade(sol, fx) -> dict:
    exact = 1.0
    worst = 0.0
    for W, g in _cases():
        ref_codes, ref_scale, ref_zero = _ref_quantize(W, g)
        try:
            got_codes, got_scale, got_zero = sol.quantize_group_affine_uint4(np.array(W, copy=True), g)
        except Exception:
            return {"exact_match": 0.0, "max_abs_err": float("inf")}

        got_codes = np.asarray(got_codes)
        got_scale = np.asarray(got_scale, dtype=np.float64)
        got_zero = np.asarray(got_zero, dtype=np.float64)

        if got_codes.shape != ref_codes.shape:
            exact = 0.0
            worst = float("inf")
            continue
        if not np.array_equal(got_codes.astype(np.int64), ref_codes.astype(np.int64)):
            exact = 0.0
        if np.any(got_codes.astype(np.int64) < 0) or np.any(got_codes.astype(np.int64) > 15):
            exact = 0.0

        try:
            recon = _ref_dequant(got_codes, got_scale, got_zero, g, W.shape)
        except Exception:
            worst = float("inf")
            continue

        ref_recon = _ref_dequant(ref_codes, ref_scale, ref_zero, g, W.shape)
        worst = max(worst, float(scorers.max_abs_err(ref_recon, recon)))

    return {"exact_match": exact, "max_abs_err": worst}
