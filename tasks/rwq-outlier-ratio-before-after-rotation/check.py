import numpy as np

from mlsys import scorers


def _hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def _ratio(X: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(X ** 2, axis=1))
    peak = np.max(np.abs(X), axis=1)
    return peak / rms


def _oracle(X: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    H = _hadamard(d)
    Xrot = X @ H.T
    return _ratio(X), _ratio(Xrot)


def _cases(rng: np.random.Generator):
    cases = []

    # small, hand-checkable: one dominant outlier channel per token
    x = np.zeros((3, 8))
    x[:, :] = rng.standard_normal((3, 8)) * 0.2
    x[0, 5] = 6.0
    x[1, 2] = -9.0
    x[2, 0] = 4.5
    cases.append(x)

    # wider batch, systematic outlier channels (different pattern from the fixture)
    n, d = 64, 32
    xb = rng.standard_normal((n, d)) * 0.25
    out_ch = rng.choice(d, size=3, replace=False)
    xb[:, out_ch] *= rng.uniform(10.0, 25.0, size=3)
    cases.append(xb)

    return cases


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    all_X = [np.asarray(fx["rot_x"], dtype=np.float64)] + _cases(rng)

    errs = []
    peak_ok = 1.0

    for X in all_X:
        exp_before, exp_after = _oracle(X)
        try:
            got_before, got_after = sol.outlier_ratio_before_after_rotation(np.array(X, copy=True))
            got_before = np.asarray(got_before, dtype=np.float64)
            got_after = np.asarray(got_after, dtype=np.float64)
        except Exception:
            errs.append(float("inf"))
            peak_ok = 0.0
            continue

        if got_before.shape != exp_before.shape or got_after.shape != exp_after.shape:
            errs.append(float("inf"))
            peak_ok = 0.0
            continue

        exp_cat = np.concatenate([exp_before, exp_after])
        got_cat = np.concatenate([got_before, got_after])
        errs.append(scorers.rel_err(exp_cat, got_cat))

        if not (np.max(got_after) < np.max(got_before)):
            peak_ok = 0.0

    return {
        "rel_err": max(errs) if errs else float("inf"),
        "peak_drops": peak_ok,
    }
