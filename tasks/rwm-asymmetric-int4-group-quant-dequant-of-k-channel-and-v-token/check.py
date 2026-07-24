import numpy as np

from mlsys import scorers


def _qd_1d(x: np.ndarray, bits: int) -> np.ndarray:
    """Asymmetric affine quant-dequant of a 1-D group, oracle formula."""
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


def _group_quant(x: np.ndarray, axis: int, group_size: int, bits: int) -> np.ndarray:
    """Quantize x in contiguous groups of group_size along `axis`; every
    slice perpendicular to `axis` gets its own independent scale/zero."""
    x = np.asarray(x, dtype=np.float64)
    rows, cols = x.shape
    out = np.empty_like(x)
    if axis == 0:
        # per-channel (column): group along the token/row axis
        for c in range(cols):
            col = x[:, c]
            for s in range(0, rows, group_size):
                seg = col[s:s + group_size]
                out[s:s + group_size, c] = _qd_1d(seg, bits)
    else:
        # per-token (row): group along the channel/col axis
        for r in range(rows):
            row = x[r]
            for s in range(0, cols, group_size):
                seg = row[s:s + group_size]
                out[r, s:s + group_size] = _qd_1d(seg, bits)
    return out


def _build_kv(seed: int, seq_len: int, channels: int):
    rng = np.random.default_rng(seed)
    K = rng.normal(loc=0.0, scale=1.0, size=(seq_len, channels))
    V = rng.normal(loc=0.0, scale=1.0, size=(seq_len, channels))
    # inject a few outliers so a wrong-axis grouping shows up clearly
    flatK = K.reshape(-1)
    idxK = rng.choice(flatK.size, size=6, replace=False)
    flatK[idxK] += rng.normal(loc=0.0, scale=8.0, size=6)
    flatV = V.reshape(-1)
    idxV = rng.choice(flatV.size, size=6, replace=False)
    flatV[idxV] += rng.normal(loc=0.0, scale=8.0, size=6)
    return K, V


def grade(sol, fx) -> dict:
    cases = [
        (0, 32, 32, 8, 4),
        (1, 64, 64, 16, 4),
        (2, 64, 32, 16, 4),
    ]

    worst_rel = 0.0
    worst_abs = 0.0

    for seed, seq_len, channels, group_size, bits in cases:
        K, V = _build_kv(seed, seq_len, channels)
        K_ref = _group_quant(K, axis=0, group_size=group_size, bits=bits)
        V_ref = _group_quant(V, axis=1, group_size=group_size, bits=bits)

        try:
            K_hat, V_hat = sol.quantize_dequantize_kv(K.copy(), V.copy(), group_size, bits=bits)
        except Exception:
            return {"rel_err": float("inf"), "max_abs_err": float("inf")}

        try:
            K_hat = np.asarray(K_hat, dtype=np.float64)
            V_hat = np.asarray(V_hat, dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf"), "max_abs_err": float("inf")}

        if K_hat.shape != K_ref.shape or V_hat.shape != V_ref.shape:
            return {"rel_err": float("inf"), "max_abs_err": float("inf")}

        ref_cat = np.concatenate([K_ref.ravel(), V_ref.ravel()])
        got_cat = np.concatenate([K_hat.ravel(), V_hat.ravel()])

        rel = scorers.rel_err(ref_cat, got_cat)
        ab = scorers.max_abs_err(ref_cat, got_cat)
        if not np.isfinite(rel):
            rel = float("inf")
        if not np.isfinite(ab):
            ab = float("inf")
        worst_rel = max(worst_rel, rel)
        worst_abs = max(worst_abs, ab)

    return {"rel_err": worst_rel, "max_abs_err": worst_abs}
