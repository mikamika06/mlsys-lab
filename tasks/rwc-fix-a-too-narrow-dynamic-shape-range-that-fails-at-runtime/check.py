import numpy as np


def _oracle(lower, upper, x):
    n = int(x.shape[0])

    new_lower = lower
    new_upper = upper

    if n < new_lower:
        new_lower = n
    if n > new_upper:
        new_upper = n

    output = np.asarray(x, dtype=np.float64)
    output = output * 2.0 + 1.0

    return (new_lower, new_upper), output


def grade(sol, fx) -> dict:
    cases = [
        (4, 6, np.arange(14, dtype=np.float32).reshape(7, 2)),
        (5, 12, np.arange(30, dtype=np.float32).reshape(15, 2)),
        (3, 9, np.arange(6, dtype=np.float64).reshape(2, 3)),
        (2, 2, np.ones((2, 4), dtype=np.int32)),
        (8, 10, np.empty((5, 3), dtype=np.float32)),
    ]

    ok = 1.0

    for lower, upper, x in cases:
        try:
            got_range, got_out = sol.fix_shape_range_and_run(lower, upper, x)
            ref_range, ref_out = _oracle(lower, upper, x)

            if tuple(got_range) != tuple(ref_range):
                ok = 0.0
                break

            got_out = np.asarray(got_out, dtype=np.float64)
            if got_out.shape != ref_out.shape:
                ok = 0.0
                break

            err = float(np.max(np.abs(got_out - ref_out))) if got_out.size else 0.0
            if err > 1e-12:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break

    return {"exact_match": ok}
