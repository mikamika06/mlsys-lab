import numpy as np


def _ref_make_qx_quants(x, w, nmax):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    amax = np.max(np.abs(x))
    if amax == 0:
        return -1, np.zeros(x.shape, dtype=np.int64)

    base_scale = amax / nmax
    best_idx = -1
    best_err = None
    best_codes = None
    for k in range(-15, 16):
        idx = k + 15
        scale = base_scale * (1.0 + k / 32.0)
        if scale == 0:
            continue
        codes = np.clip(np.round(x / scale), -nmax, nmax)
        err = float(np.sum(w * (x - scale * codes) ** 2))
        if best_err is None or err < best_err:
            best_err = err
            best_idx = idx
            best_codes = codes.astype(np.int64)
    return best_idx, best_codes


def _scenarios():
    rng = np.random.default_rng(0)
    scenarios = []

    # uniform weights
    x = rng.normal(size=16)
    w = np.ones(16)
    scenarios.append((x, w, 7))

    # sharply skewed weights on one element
    x = rng.normal(size=12)
    w = np.full(12, 0.1)
    w[3] = 50.0
    scenarios.append((x, w, 7))

    # all-zero block
    scenarios.append((np.zeros(8), np.ones(8), 7))

    # different nmax (q8-like)
    x = rng.normal(size=32) * 5
    w = np.abs(rng.normal(size=32)) + 0.01
    scenarios.append((x, w, 127))

    # small nmax, wide weight spread
    x = rng.uniform(-3, 3, size=10)
    w = rng.uniform(0.01, 20.0, size=10)
    scenarios.append((x, w, 3))

    # negative-heavy values
    x = -np.abs(rng.normal(size=20)) * 2
    w = rng.uniform(0.1, 5.0, size=20)
    scenarios.append((x, w, 7))

    return scenarios


def grade(sol, fx) -> dict:
    total = 0
    correct = 0

    for x, w, nmax in _scenarios():
        total += 1
        idx_ref, codes_ref = _ref_make_qx_quants(x, w, nmax)

        try:
            idx_got, codes_got = sol.make_qx_quants(x.copy(), w.copy(), nmax)
        except Exception:
            continue

        try:
            idx_got = int(idx_got)
            codes_got = np.asarray(codes_got)
        except Exception:
            continue

        if codes_got.shape != codes_ref.shape:
            continue
        if idx_got != idx_ref:
            continue
        if not np.array_equal(codes_got.astype(np.int64), codes_ref):
            continue

        correct += 1

    argmin_index = (correct / total) if total else 0.0
    return {"argmin_index": argmin_index}
