import numpy as np

def _ref(timestamps, latencies, admitted, slo_threshold, window):
    """NumPy oracle for goodput."""
    mask = np.asarray(admitted, dtype=bool) & (
        np.asarray(latencies, dtype=np.float64) <= slo_threshold
    )
    return float(np.sum(mask) / window)

def grade(sol, fx) -> dict:
    cases = [
        # All admitted, mixed latencies vs SLO
        (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0.1, 0.5, 0.3, 0.2, 0.8]),
            np.array([True, True, True, True, True]),
            0.4, 5.0,
        ),
        # Half admitted — catches students who ignore the admitted mask
        (
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.1, 0.2, 0.15, 0.3]),
            np.array([True, False, True, False]),
            0.25, 2.0,
        ),
        # Single request, exceeds SLO
        (
            np.array([0.0]),
            np.array([1.0]),
            np.array([True]),
            0.5, 1.0,
        ),
        # All meet SLO
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.1, 0.1, 0.1]),
            np.array([True, True, True]),
            1.0, 3.0,
        ),
        # No requests admitted
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.05, 0.05, 0.05]),
            np.array([False, False, False]),
            0.1, 3.0,
        ),
        # Larger trace with 10 requests
        (
            np.linspace(0.0, 9.0, 10),
            np.array([0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.15, 0.35, 0.55, 0.25]),
            np.array([True, True, False, True, True, False, True, True, False, True]),
            0.3, 10.0,
        ),
    ]

    ok = 1.0
    for ts, lat, adm, slo, win in cases:
        try:
            got = float(sol.compute_goodput(ts, lat, adm, slo, win))
        except Exception:
            ok = 0.0
            break
        expected = _ref(ts, lat, adm, slo, win)
        if abs(got - expected) > 1e-12:
            ok = 0.0
            break
    return {"exact_match": ok}
