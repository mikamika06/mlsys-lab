import numpy as np
from mlsys import probe


def _oracle(logits, T):
    """Reference temperature-scaled stable softmax, computed with NumPy in float64."""
    z = np.asarray(logits, dtype=np.float64) / T
    m = np.max(z)
    e = np.exp(z - m)
    return e / np.sum(e)


def grade(sol, fx) -> dict:
    # Correctness cases. Several deliberately overflow a naive exp(logits / T):
    # large raw logits, and small temperatures that blow the scaled logits up to
    # ~350. Only a max-shifted (stable) reduction survives them.
    rng = np.random.default_rng(0)
    cases = [
        (np.array([1.0, 2.0, 3.0]), 1.0),
        (np.array([1000.0, 1001.0, 1002.0]), 1.0),        # raw overflow
        (np.array([30.0, 32.0, 35.0]), 0.1),              # temperature-amplified overflow
        (np.array([1.0, 2.0, 3.0, 4.0]), 5.0),            # smoothing (T > 1)
        (np.array([-1000.0, -999.0, -998.0]), 1.0),       # large negatives
        (np.array([5.0]), 2.0),                           # single element
        (np.array([2.0, 2.0, 2.0]), 1.0),                 # ties -> uniform
        (rng.standard_normal(10), 0.7),                   # random moderate
    ]

    worst = 0.0
    try:
        for logits, T in cases:
            got = np.asarray(sol.fused_softmax(logits, T), dtype=np.float64)
            ref = _oracle(logits, T)
            if got.shape != ref.shape or not np.all(np.isfinite(got)):
                return {"max_abs_err": float("inf"), "line_events": float("inf")}
            err = float(np.max(np.abs(got - ref)))
            worst = max(worst, err)
    except Exception:
        return {"max_abs_err": float("inf"), "line_events": float("inf")}

    # Single-pass probe: count Python line events during one call on a fixed
    # 256-element row. An online single pass touches each element once (~5 events
    # per element here); a naive max / exp+sum / normalize triple loop over the row
    # roughly triples that and blows past the gate.
    probe_logits = rng.standard_normal(256)
    probe_T = 0.7
    try:
        events = probe.count_line_events(sol.fused_softmax, probe_logits, probe_T)
    except Exception:
        return {"max_abs_err": worst, "line_events": float("inf")}

    return {"max_abs_err": worst, "line_events": float(events)}
