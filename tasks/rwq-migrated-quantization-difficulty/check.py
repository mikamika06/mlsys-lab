import numpy as np

from mlsys import scorers

_EPS = 1e-12


def _oracle(X, W, alpha):
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    amax_X = np.max(np.abs(X), axis=0)
    amax_W = np.max(np.abs(W), axis=0)

    ratio_before = amax_X / max(float(np.mean(amax_X)), _EPS)

    s = (np.maximum(amax_X, _EPS) ** alpha) / (np.maximum(amax_W, _EPS) ** (1.0 - alpha))
    amax_X_smoothed = amax_X / s
    ratio_after = amax_X_smoothed / max(float(np.mean(amax_X_smoothed)), _EPS)

    return ratio_before, ratio_after


def _make_case(seed, n_tok, C, n_outlier, outlier_mult, alpha):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_tok, C)).astype(np.float64)
    W = rng.normal(size=(8, C)).astype(np.float64) * 0.1
    outlier_idx = rng.choice(C, size=n_outlier, replace=False)
    X[:, outlier_idx] *= outlier_mult
    return X, W, alpha


def _cases():
    return [
        _make_case(1, 64, 16, 2, 50.0, 0.5),
        _make_case(2, 128, 32, 3, 30.0, 0.5),
        _make_case(3, 40, 8, 1, 100.0, 0.3),
        _make_case(4, 200, 64, 5, 20.0, 0.75),
    ]


def grade(sol, fx) -> dict:
    worst_rel = 0.0
    peak_ok = 1.0

    for X, W, alpha in _cases():
        ref_before, ref_after = _oracle(X, W, alpha)
        try:
            got_before, got_after = sol.channel_peakiness_before_after(
                np.array(X, copy=True), np.array(W, copy=True), alpha
            )
            got_before = np.asarray(got_before, dtype=np.float64)
            got_after = np.asarray(got_after, dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf"), "peak_drop_ok": 0.0}

        if got_before.shape != ref_before.shape or got_after.shape != ref_after.shape:
            return {"rel_err": float("inf"), "peak_drop_ok": 0.0}
        if not (np.all(np.isfinite(got_before)) and np.all(np.isfinite(got_after))):
            return {"rel_err": float("inf"), "peak_drop_ok": 0.0}

        combined_ref = np.concatenate([ref_before, ref_after])
        combined_got = np.concatenate([got_before, got_after])
        worst_rel = max(worst_rel, scorers.rel_err(combined_ref, combined_got))

        if not (float(np.max(got_after)) < float(np.max(got_before))):
            peak_ok = 0.0

    return {"rel_err": worst_rel, "peak_drop_ok": peak_ok}
