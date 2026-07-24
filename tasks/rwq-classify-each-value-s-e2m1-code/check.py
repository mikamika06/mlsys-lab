import numpy as np

GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _oracle(x):
    x = np.asarray(x, dtype=np.float64)
    absx = np.abs(x)
    dist = np.abs(absx[..., None] - GRID)
    idx = np.argmin(dist, axis=-1)
    sign_bit = (x < 0).astype(np.int64)
    return (sign_bit * 8 + idx).astype(np.int64)


def grade(sol, fx) -> dict:
    cases = [np.asarray(fx["fp4_x"], dtype=np.float64)]

    rng = np.random.default_rng(7)
    cases.append(rng.uniform(-8.0, 8.0, size=(5, 17)))
    cases.append(np.array([0.0, -0.0, 6.0, -6.0, 3.0, -3.0]))

    ok = 1.0
    for x in cases:
        ref = _oracle(x)
        try:
            got = np.asarray(sol.e2m1_classify(x.copy()))
        except Exception:
            return {"exact_match": 0.0}

        if got.shape != ref.shape:
            ok = 0.0
            break
        if not np.array_equal(got.astype(np.int64), ref):
            ok = 0.0
            break

    return {"exact_match": ok}
