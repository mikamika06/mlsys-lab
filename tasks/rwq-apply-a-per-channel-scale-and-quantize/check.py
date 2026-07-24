import numpy as np

from mlsys import scorers


def _qd_1d(x: np.ndarray, bits: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return x.copy()
    scale = (xmax - xmin) / qmax
    zero = round(-xmin / scale)
    zero = min(max(zero, 0), qmax)
    codes = np.clip(np.round(x / scale) + zero, 0, qmax)
    return (codes - zero) * scale


def _group_quant_rows(W: np.ndarray, group_size: int, bits: int) -> np.ndarray:
    """Per-output-row (per-out-channel) int-`bits` group-affine quant,
    grouped along the in_features axis (axis=1)."""
    W = np.asarray(W, dtype=np.float64)
    rows, cols = W.shape
    out = np.empty_like(W)
    for r in range(rows):
        row = W[r]
        for c0 in range(0, cols, group_size):
            seg = row[c0:c0 + group_size]
            out[r, c0:c0 + group_size] = _qd_1d(seg, bits)
    return out


def _build_cases():
    cases = []
    for seed, out_f, in_f, group_size in [(0, 16, 64, 16), (1, 12, 48, 24), (2, 8, 32, 32)]:
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(out_f, in_f))
        X = rng.normal(size=(6, in_f))
        s = rng.uniform(0.5, 3.0, size=(in_f,))
        cases.append((W, X, s, group_size, 4))
    return cases


def grade(sol, fx) -> dict:
    worst_identity = 0.0
    worst_quant = 0.0

    for W, X, s, group_size, bits in _build_cases():
        Y_direct = X @ W.T
        Wp_ref = W * s[None, :]
        W_hat_ref = _group_quant_rows(Wp_ref, group_size, bits)
        Xp_ref = X / s[None, :]
        Y_quant_ref = Xp_ref @ W_hat_ref.T

        try:
            got = sol.awq_scale_and_quantize(W.copy(), X.copy(), s.copy(), group_size, bits=bits)
        except Exception:
            return {"identity_max_abs_err": float("inf"), "quant_max_abs_err": float("inf")}

        try:
            Y_id, Y_q = got
            Y_id = np.asarray(Y_id, dtype=np.float64)
            Y_q = np.asarray(Y_q, dtype=np.float64)
        except Exception:
            return {"identity_max_abs_err": float("inf"), "quant_max_abs_err": float("inf")}

        if Y_id.shape != Y_direct.shape or Y_q.shape != Y_quant_ref.shape:
            return {"identity_max_abs_err": float("inf"), "quant_max_abs_err": float("inf")}

        id_err = scorers.max_abs_err(Y_direct, Y_id)
        q_err = scorers.max_abs_err(Y_quant_ref, Y_q)
        if not np.isfinite(id_err):
            id_err = float("inf")
        if not np.isfinite(q_err):
            q_err = float("inf")
        worst_identity = max(worst_identity, id_err)
        worst_quant = max(worst_quant, q_err)

    return {"identity_max_abs_err": worst_identity, "quant_max_abs_err": worst_quant}
