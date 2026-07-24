import numpy as np

from mlsys import scorers

QK = 32


def _oracle(w: np.ndarray):
    """Exact port of ggml's quantize_row_q4_1_reference."""
    w = np.asarray(w, dtype=np.float64)
    n_blocks = w.shape[0]
    mn = w.min(axis=1)
    mx = w.max(axis=1)
    d = (mx - mn) / 15.0
    inv_d = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0)

    x0 = (w - mn[:, None]) * inv_d[:, None]
    codes = np.minimum(15, np.floor(x0 + 0.5)).astype(np.uint8)
    codes = np.clip(codes, 0, 15)

    d16 = np.float16(d).astype(np.float64)
    m16 = np.float16(mn).astype(np.float64)
    return d16, m16, codes


def _cases():
    rng = np.random.default_rng(4111)
    blocks = []

    blocks.append(rng.normal(size=(5, QK)))
    blocks.append(rng.uniform(-50, 50, size=(4, QK)))
    blocks.append(rng.normal(scale=0.01, size=(3, QK)))

    const_block = np.full((2, QK), 3.5)
    blocks.append(const_block)

    mixed = rng.normal(size=(6, QK))
    mixed[0] = 7.0  # one constant row mixed in with varying ones
    blocks.append(mixed)

    return [b.astype(np.float64) for b in blocks]


def grade(sol, fx) -> dict:
    worst_code_frac = 1.0
    worst_dm_err = 0.0

    for w in _cases():
        d_ref, m_ref, codes_ref = _oracle(w)
        try:
            d, m, codes = sol.q4_1_quantize(w.copy())
            d = np.asarray(d, dtype=np.float64)
            m = np.asarray(m, dtype=np.float64)
            codes = np.asarray(codes, dtype=np.uint8)
        except Exception:
            return {"byte_exact_fraction": 0.0, "max_abs_err": float("inf")}

        if d.shape != d_ref.shape or m.shape != m_ref.shape or codes.shape != codes_ref.shape:
            return {"byte_exact_fraction": 0.0, "max_abs_err": float("inf")}
        if not (np.all(np.isfinite(d)) and np.all(np.isfinite(m))):
            return {"byte_exact_fraction": 0.0, "max_abs_err": float("inf")}
        if codes.max(initial=0) > 15:
            return {"byte_exact_fraction": 0.0, "max_abs_err": float("inf")}

        frac = scorers.byte_exact_fraction(codes_ref, codes)
        worst_code_frac = min(worst_code_frac, frac)

        dm_err = max(
            float(scorers.max_abs_err(d_ref, d)),
            float(scorers.max_abs_err(m_ref, m)),
        )
        worst_dm_err = max(worst_dm_err, dm_err)

    return {"byte_exact_fraction": worst_code_frac, "max_abs_err": worst_dm_err}
