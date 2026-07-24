import numpy as np


def _oracle(vocab_size, trace, allowed):
    densities = []
    for state in trace:
        allowed_count = len(set(allowed[state]))
        densities.append((vocab_size - allowed_count) / vocab_size)
    arr = np.asarray(densities, dtype=np.float64)
    return arr, float(np.mean(arr))


def _rel_err(a, b):
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-12))


def grade(sol, fx) -> dict:
    cases = [
        (
            16,
            [0, 1, 2, 1, 0],
            {
                0: [1, 4, 7],
                1: [0, 2, 3, 5, 8],
                2: [9, 10, 11, 12, 13, 14, 15],
            },
        ),
        (
            32,
            [3, 3, 0, 2, 1],
            {
                0: list(range(0, 8)),
                1: [2, 4, 6],
                2: [1, 3, 5, 7, 9, 11],
                3: list(range(20)),
            },
        ),
        (
            100,
            [5, 4, 5],
            {
                4: [0, 1, 2, 3, 4],
                5: list(range(50)),
            },
        ),
    ]

    errors = []
    for vocab_size, trace, allowed in cases:
        try:
            got_d, got_m = sol.mask_density_trace(
                vocab_size, list(trace), dict(allowed)
            )
        except Exception:
            return {"rel_err": float("inf")}

        ref_d, ref_m = _oracle(vocab_size, trace, allowed)
        errors.append(_rel_err(got_d, ref_d))
        errors.append(_rel_err([got_m], [ref_m]))

    return {"rel_err": float(max(errors))}
