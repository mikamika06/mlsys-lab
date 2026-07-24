import numpy as np

from mlsys import scorers


def _wmse(x, x_hat, w):
    return float(np.sum(w * (x - x_hat) ** 2) / np.sum(w))


def _q4_0(x):
    amax = np.max(np.abs(x))
    d = amax / 8.0 if amax != 0 else 1e-12
    codes = np.clip(np.round(x / d), -8, 7)
    return d * codes


def _search_scale(x, weight):
    amax = np.max(np.abs(x))
    d0 = amax / 8.0 if amax != 0 else 1e-12
    best_err = None
    best_recon = None
    for k in range(-15, 16):
        d = d0 * (1.0 + k / 32.0)
        if d == 0:
            continue
        codes = np.clip(np.round(x / d), -8, 7)
        recon = d * codes
        err = float(np.sum(weight * (x - recon) ** 2))
        if best_err is None or err < best_err:
            best_err = err
            best_recon = recon
    return best_recon


def _oracle(x, w):
    recon_q4_0 = _q4_0(x)
    recon_q4_k = _search_scale(x, np.ones_like(w))
    recon_imatrix = _search_scale(x, w)
    errors = np.array(
        [_wmse(x, recon_q4_0, w), _wmse(x, recon_q4_k, w), _wmse(x, recon_imatrix, w)],
        dtype=np.float64,
    )
    best_idx = int(np.argmin(errors))
    return errors, best_idx


def _scenarios():
    rng = np.random.default_rng(0)
    scenarios = []

    # ordinary calibration-like row, uniform weights
    scenarios.append((rng.normal(size=32), np.ones(32)))

    # sharply skewed weights concentrating importance on one element
    x = rng.normal(size=32) * 2.0
    w = np.full(32, 0.05)
    w[7] = 40.0
    scenarios.append((x, w))

    # all-equal-magnitude block (edge case for the fixed scale)
    x = rng.choice([-3.0, 3.0], size=32)
    w = rng.uniform(0.1, 5.0, size=32)
    scenarios.append((x, w))

    # wide-range calibration-like block with a couple of outliers
    x = rng.normal(size=32)
    x[2] = 15.0
    x[19] = -12.0
    w = np.abs(rng.normal(size=32)) + 0.01
    scenarios.append((x, w))

    # smoothly varying weights, small values
    x = rng.uniform(-0.5, 0.5, size=32)
    w = np.linspace(0.1, 10.0, 32)
    scenarios.append((x, w))

    # negative-heavy block
    x = -np.abs(rng.normal(size=32)) * 3
    w = rng.uniform(0.1, 8.0, size=32)
    scenarios.append((x, w))

    return scenarios


def grade(sol, fx) -> dict:
    got_all = []
    ref_all = []
    matches = 0
    total = 0

    for x, w in _scenarios():
        total += 1
        ref_errors, ref_idx = _oracle(x, w)

        try:
            got_errors, got_idx = sol.compare_q4_variants(x.copy(), w.copy())
            got_errors = np.asarray(got_errors, dtype=np.float64)
            got_idx = int(got_idx)
        except Exception:
            return {"rel_err": float("inf"), "argmin_match": 0.0}

        if got_errors.shape != ref_errors.shape or not np.all(np.isfinite(got_errors)):
            return {"rel_err": float("inf"), "argmin_match": 0.0}

        got_all.append(got_errors)
        ref_all.append(ref_errors)
        if got_idx == ref_idx:
            matches += 1

    got_stack = np.concatenate(got_all)
    ref_stack = np.concatenate(ref_all)

    return {
        "rel_err": float(scorers.rel_err(ref_stack, got_stack)),
        "argmin_match": (matches / total) if total else 0.0,
    }
