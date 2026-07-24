import numpy as np
from mlsys import probe

N = 3000


def _make_array():
    """A shuffled array engineered so naive float32 accumulation loses most
    of the small increments: a huge value forces the running sum's ULP well
    above 1.0, so `running_sum + 1.0` frequently rounds back to
    `running_sum` in plain float32 arithmetic."""
    rng = np.random.RandomState(42)
    arr = np.array([1e10] + [1.0] * N + [-1e10], dtype=np.float32)
    idx = rng.permutation(len(arr))
    return arr[idx]


def _naive_fp32_sum(arr):
    s = np.float32(0.0)
    for x in arr:
        s = np.float32(s + x)
    return s


def grade(sol, fx) -> dict:
    arr = _make_array()
    ref = float(np.sum(arr.astype(np.float64)))

    naive = float(_naive_fp32_sum(arr))
    naive_rel_err = abs(naive - ref) / (abs(ref) + 1e-15)

    try:
        n_events = probe.count_line_events(sol.compensated_sum, arr.copy())
        got = float(sol.compensated_sum(arr.copy()))
    except Exception:
        return {"improvement_ratio": 0.0, "loop_events": 0.0}

    student_rel_err = abs(got - ref) / (abs(ref) + 1e-15)
    ratio = naive_rel_err / max(student_rel_err, 1e-15)

    return {
        "improvement_ratio": float(ratio),
        "loop_events": float(n_events),
    }
