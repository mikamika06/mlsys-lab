import numpy as np


def _oracle(chunks):
    full = np.concatenate([np.asarray(c, dtype=np.float64) for c in chunks], axis=1)
    m = np.max(full, axis=1)
    return m + np.log(np.sum(np.exp(full - m[:, None]), axis=1))


def _split(full, widths):
    out = []
    i = 0
    for w in widths:
        out.append(full[:, i:i + w])
        i += w
    return out


def _run_checked(sol, chunks):
    """Call sol.streaming_logsumexp(chunks), flagging a violation if np.exp is
    ever invoked on an array wider than the widest individual chunk given --
    i.e. the chunks were secretly concatenated before exponentiating."""
    max_w = max(c.shape[1] for c in chunks)
    orig_exp = np.exp
    violation = [False]

    def traced_exp(x, *a, **kw):
        arr = np.asarray(x)
        # only 2D (rows x columns) exponentiations are subject to the
        # per-chunk-width limit; 1D per-row rescale factors (e.g. exp(m_old
        # - m_new)) are not column data and don't count.
        if arr.ndim == 2 and arr.shape[1] > max_w:
            violation[0] = True
        return orig_exp(x, *a, **kw)

    np.exp = traced_exp
    try:
        out = sol.streaming_logsumexp([c.copy() for c in chunks])
    finally:
        np.exp = orig_exp
    return out, violation[0]


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    cases = []
    full1 = rng.standard_normal((20, 24)) * 5.0
    cases.append(_split(full1, [12, 12]))

    full2 = rng.standard_normal((15, 30)) * 3.0 + rng.standard_normal((15, 1)) * 10.0
    cases.append(_split(full2, [6, 6, 6, 6, 6]))

    full3 = rng.standard_normal((5, 10)) * 2.0
    cases.append(_split(full3, [10]))  # single chunk edge case

    worst_err = 0.0
    streamed_ok = 1.0

    for chunks in cases:
        ref = _oracle(chunks)
        try:
            got, violation = _run_checked(sol, chunks)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"), "streamed": 0.0}

        if violation:
            streamed_ok = 0.0

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf"), "streamed": 0.0}

        err = float(np.max(np.abs(got - ref)))
        worst_err = max(worst_err, err)

    return {"max_abs_err": worst_err, "streamed": streamed_ok}
