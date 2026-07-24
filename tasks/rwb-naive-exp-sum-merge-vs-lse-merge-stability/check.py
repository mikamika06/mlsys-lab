import numpy as np


def _oracle_lse_merge(chunk_scores, chunk_values):
    ms, ss, os_ = [], [], []
    for L, V in zip(chunk_scores, chunk_values):
        m = float(np.max(L))
        e = np.exp(L - m)
        ms.append(m)
        ss.append(float(np.sum(e)))
        os_.append((e[:, None] * V).sum(axis=0))
    ms = np.array(ms, dtype=np.float64)
    ss = np.array(ss, dtype=np.float64)
    os_ = np.array(os_, dtype=np.float64)

    gmax = float(np.max(ms))
    alpha = np.exp(ms - gmax)
    total_sumexp = float(np.sum(alpha * ss))
    total_output = (alpha[:, None] * os_).sum(axis=0)
    return total_output / total_sumexp


def _oracle_naive_merge(chunk_scores, chunk_values):
    num = None
    den = 0.0
    for L, V in zip(chunk_scores, chunk_values):
        e = np.exp(L)
        contrib = (e[:, None] * V).sum(axis=0)
        num = contrib if num is None else num + contrib
        den += float(np.sum(e))
    return num / den


def _make_small_case(rng, d):
    chunk_scores = [rng.standard_normal(6) * 2.0 for _ in range(3)]
    chunk_values = [rng.standard_normal((6, d)) for _ in range(3)]
    return chunk_scores, chunk_values


def _make_large_case(rng, d):
    """Scores large enough that exp() overflows float64 (>~709): the naive
    merge's numerator AND denominator both become +-inf, so the ratio is
    NaN -- exactly the failure mode the LSE merge is designed to avoid."""
    chunk_scores = [rng.uniform(700.0, 950.0, size=6) for _ in range(3)]
    chunk_values = [rng.standard_normal((6, d)) for _ in range(3)]
    return chunk_scores, chunk_values


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(5)
    d = 4

    with np.errstate(over="ignore", invalid="ignore"):
        # --- lse_err: must stay accurate at BOTH ordinary and huge magnitude ---
        worst_lse_err = 0.0
        for maker in (_make_small_case, _make_large_case):
            chunk_scores, chunk_values = maker(rng, d)
            ref = _oracle_lse_merge(chunk_scores, chunk_values)
            try:
                got = np.asarray(
                    sol.lse_merge([c.copy() for c in chunk_scores], [v.copy() for v in chunk_values]),
                    dtype=np.float64,
                )
            except Exception:
                return {"lse_err": float("inf"), "naive_check": 0.0}
            if got.shape != ref.shape or not np.all(np.isfinite(got)):
                return {"lse_err": float("inf"), "naive_check": 0.0}
            worst_lse_err = max(worst_lse_err, float(np.max(np.abs(got - ref))))

        # --- naive_check: correct (unstable) formula reproduced in BOTH regimes ---
        naive_ok = 1.0

        small_scores, small_values = _make_small_case(rng, d)
        ref_small = _oracle_naive_merge(small_scores, small_values)
        try:
            got_small = np.asarray(
                sol.naive_merge([c.copy() for c in small_scores], [v.copy() for v in small_values]),
                dtype=np.float64,
            )
        except Exception:
            return {"lse_err": worst_lse_err, "naive_check": 0.0}
        if got_small.shape != ref_small.shape or np.max(np.abs(got_small - ref_small)) > 1e-6:
            naive_ok = 0.0

        big_scores, big_values = _make_large_case(rng, d)
        ref_big = _oracle_naive_merge(big_scores, big_values)  # expected: all-NaN
        try:
            got_big = np.asarray(
                sol.naive_merge([c.copy() for c in big_scores], [v.copy() for v in big_values]),
                dtype=np.float64,
            )
        except Exception:
            return {"lse_err": worst_lse_err, "naive_check": 0.0}
        if got_big.shape != ref_big.shape:
            naive_ok = 0.0
        elif not (np.all(np.isnan(ref_big)) and np.all(np.isnan(got_big))):
            naive_ok = 0.0

    return {"lse_err": worst_lse_err, "naive_check": naive_ok}
