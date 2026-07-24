import sys
import numpy as np


def _oracle(x, window):
    view = np.lib.stride_tricks.sliding_window_view(
        np.asarray(x, dtype=np.float64), window_shape=window
    )
    return np.mean(view, axis=1, dtype=np.float64)


def _line_events(sol, x, window):
    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        if event == "line":
            count += 1
        return tracer

    old = sys.gettrace()
    try:
        sys.settrace(tracer)
        sol.rolling_window_mean(x, window)
    except Exception:
        return 10_000
    finally:
        sys.settrace(old)
    return count


def grade(sol, fx) -> dict:
    cases = [
        (np.arange(20, dtype=np.float64), 4),
        (np.sin(np.linspace(-3, 3, 101)), 7),
        (np.array([3.5, -1.2, 8.0, 4.4, 0.0, 2.1]), 3),
        (np.linspace(0.1, 1.0, 257), 16),
    ]

    worst = 0.0
    for x, window in cases:
        try:
            got = np.asarray(sol.rolling_window_mean(x, window), dtype=np.float64)
        except Exception:
            return {"rel_err": 1.0}

        ref = _oracle(x, window)
        if got.shape != ref.shape:
            return {"rel_err": 1.0}

        err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
        worst = max(worst, float(err))

    # A real execution trace guard: a window-by-window Python implementation
    # creates many line events on larger inputs.
    if _line_events(sol, np.arange(1000, dtype=np.float64), 32) > 80:
        worst = max(worst, 1.0)

    return {"rel_err": worst}
