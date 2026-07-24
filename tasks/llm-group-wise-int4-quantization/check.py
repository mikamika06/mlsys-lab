import numpy as np
from mlsys import scorers


def _ref_quantize(W, group_size):
    flat = np.asarray(W, dtype=np.float64).ravel()
    q = np.empty(flat.shape, dtype=np.int8)
    scales = []
    for start in range(0, len(flat), group_size):
        group = flat[start:start + group_size]
        mx = np.max(np.abs(group))
        scale = 1.0 if mx == 0 else mx / 7.0
        scales.append(scale)
        vals = np.rint(group / scale)
        vals = np.clip(vals, -8, 7)
        q[start:start + len(group)] = vals.astype(np.int8)
    return q, np.asarray(scales, dtype=np.float64)


def _dequant(q, scales, group_size, shape):
    out = np.empty(len(q), dtype=np.float64)
    for i, scale in enumerate(scales):
        start = i * group_size
        end = min(len(q), start + group_size)
        out[start:end] = q[start:end].astype(np.float64) * scale
    return out.reshape(shape)


def grade(sol, fx) -> dict:
    cases = [
        (np.array([[0.0, 1.5, -2.5, 7.0]], dtype=np.float64), 2),
        (np.array([[3.2, -1.1, 0.0], [4.5, -8.0, 2.0]], dtype=np.float64), 3),
        (np.arange(-17, 18, dtype=np.float64).reshape(7, 5) / 3.0, 8),
        (np.zeros((2, 6), dtype=np.float64), 4),
    ]

    exact = 1.0
    err = 0.0
    for W, g in cases:
        ref_q, ref_s = _ref_quantize(W, g)
        try:
            got_q, got_s, got_shape = sol.quantize_groupwise_int4(W, g)
        except Exception:
            return {"rel_err": 1.0, "exact_match": 0.0}

        if tuple(got_shape) != tuple(W.shape):
            exact = 0.0
        if not np.array_equal(np.asarray(got_q), ref_q):
            exact = 0.0
        if not np.array_equal(np.asarray(got_s), ref_s):
            exact = 0.0

        try:
            recon = _dequant(np.asarray(got_q), np.asarray(got_s), g, tuple(got_shape))
            err = max(err, scorers.rel_err(W, recon))
        except Exception:
            err = 1.0

    return {"rel_err": float(err), "exact_match": float(exact)}
