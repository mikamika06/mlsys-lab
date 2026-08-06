import numpy as np

from mlsys import scorers


def _log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=-1, keepdims=True)
    s = z - m
    return s - np.log(np.sum(np.exp(s), axis=-1, keepdims=True))


def _loss(logits, labels):
    """Mean cross-entropy, computed stably in float64 (the oracle forward)."""
    lp = _log_softmax(logits)
    n = len(labels)
    return float(-np.mean(lp[np.arange(n), labels]))


def _ref_grad(logits, labels):
    """Analytic oracle: dL/dlogits = (softmax(logits) - onehot(labels)) / N."""
    p = np.exp(_log_softmax(logits))
    n = len(labels)
    p[np.arange(n), labels] -= 1.0
    return p / n


def _fd_grad(logits, labels, eps=1e-5):
    """Central finite differences on the oracle forward — an independent check."""
    g = np.zeros_like(logits, dtype=np.float64)
    it = np.ndindex(*logits.shape)
    for idx in it:
        h = eps * max(1.0, abs(float(logits[idx])))
        up = logits.copy()
        dn = logits.copy()
        up[idx] += h
        dn[idx] -= h
        g[idx] = (_loss(up, labels) - _loss(dn, labels)) / (2.0 * h)
    return g


def _cases(rng):
    cases = []
    # 1. ordinary logits
    cases.append((rng.standard_normal((6, 7)), rng.integers(0, 7, size=6)))
    # 2. huge magnitudes -> naive exp() overflows, a stable log-softmax does not
    cases.append((rng.standard_normal((5, 4)) * 400.0, rng.integers(0, 4, size=5)))
    # 3. one near-deterministic row plus a flat row
    z = rng.standard_normal((4, 5)) * 0.01
    z[0] += np.array([0.0, 0.0, 900.0, 0.0, 0.0])
    z[1] = 0.0
    cases.append((z, np.array([2, 3, 0, 4])))
    # 4. large negative offset (shift invariance)
    cases.append((rng.standard_normal((3, 6)) - 1e3, rng.integers(0, 6, size=3)))
    return [(a.tolist(), b.tolist()) for a, b in cases]


def _fail():
    return {"max_abs_err": float("inf"), "fd_max_abs_err": float("inf"), "sum_zero_err": float("inf")}


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    worst = 0.0
    worst_sum = 0.0
    for logits, labels in _cases(rng):
        ref = _ref_grad(np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64))
        try:
            got = sol.cross_entropy_backward([row[:] for row in logits], list(labels))
        except Exception:
            return _fail()
        try:
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return _fail()
        if got.shape != np.shape(logits) or not np.all(np.isfinite(got)):
            return _fail()
        worst = max(worst, scorers.max_abs_err(ref, got))
        # rows of the gradient must sum to 0 (softmax sums to 1, onehot sums to 1)
        worst_sum = max(worst_sum, float(np.max(np.abs(np.sum(got, axis=-1)))))

    # independent oracle: central finite differences of the loss
    fd_logits = rng.standard_normal((4, 5))
    fd_labels = rng.integers(0, 5, size=4).astype(np.int64)
    try:
        got_fd = np.asarray(sol.cross_entropy_backward(fd_logits.tolist(), fd_labels.tolist()), dtype=np.float64)
    except Exception:
        return _fail()
    if got_fd.shape != fd_logits.shape or not np.all(np.isfinite(got_fd)):
        return _fail()
    fd_ref = _fd_grad(fd_logits, fd_labels)
    fd_err = scorers.max_abs_err(fd_ref, got_fd)

    return {
        "max_abs_err": float(worst),
        "fd_max_abs_err": float(fd_err),
        "sum_zero_err": float(worst_sum),
    }
