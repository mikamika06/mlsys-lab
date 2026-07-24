import numpy as np


def _oracle(attention, window_size, budget, needle_index):
    history = np.asarray(attention, dtype=np.float64)
    n = history.shape[1]
    mass = np.sum(history, axis=0)

    stream = set(range(min(2, n)))
    start = max(0, n - window_size)
    stream.update(range(start, n))
    stream = sorted(stream)

    order = sorted(range(n), key=lambda i: (-mass[i], i))
    h2o = sorted(order[:min(budget, n)])

    return {
        "streaming_retained": stream,
        "h2o_retained": h2o,
        "streaming_keeps_needle": needle_index in stream,
        "h2o_keeps_needle": needle_index in h2o,
        "streaming_mass": float(np.sum(mass[stream])),
        "h2o_mass": float(np.sum(mass[h2o])),
    }


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([
                [0.1, 0.1, 0.1, 0.1, 0.6],
                [0.1, 0.1, 0.1, 0.6, 0.1],
                [0.1, 0.1, 0.8, 0.05, 0.05],
            ]),
            2,
            3,
            2,
        ),
        (
            np.array([
                [0.4, 0.2, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.05, 0.025, 0.025],
            ]),
            2,
            2,
            2,
        ),
        (
            np.array([
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ]),
            1,
            2,
            3,
        ),
    ]

    ok = 1.0
    for attention, window_size, budget, needle_index in cases:
        expected = _oracle(attention, window_size, budget, needle_index)
        try:
            got = sol.compare_retention(attention, window_size, budget, needle_index)
            got = {
                "streaming_retained": list(got["streaming_retained"]),
                "h2o_retained": list(got["h2o_retained"]),
                "streaming_keeps_needle": bool(got["streaming_keeps_needle"]),
                "h2o_keeps_needle": bool(got["h2o_keeps_needle"]),
                "streaming_mass": float(got["streaming_mass"]),
                "h2o_mass": float(got["h2o_mass"]),
            }
        except Exception:
            ok = 0.0
            break

        if got["streaming_retained"] != expected["streaming_retained"]:
            ok = 0.0
            break
        if got["h2o_retained"] != expected["h2o_retained"]:
            ok = 0.0
            break
        if got["streaming_keeps_needle"] != expected["streaming_keeps_needle"]:
            ok = 0.0
            break
        if got["h2o_keeps_needle"] != expected["h2o_keeps_needle"]:
            ok = 0.0
            break
        if not np.isclose(got["streaming_mass"], expected["streaming_mass"]):
            ok = 0.0
            break
        if not np.isclose(got["h2o_mass"], expected["h2o_mass"]):
            ok = 0.0
            break

    return {"exact_match": ok}
