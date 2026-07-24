import numpy as np

from mlsys import scorers


def _baseline(A, B, C):
    D = (A + B) * C
    E = np.maximum(D, 0.0) - A
    F = float(np.sum(E))
    return E, F


def _gen_case(rng):
    n = int(rng.integers(1, 40))
    tile_size = int(rng.integers(1, 12))
    A = rng.standard_normal(n)
    B = rng.standard_normal(n)
    C = rng.standard_normal(n)
    return A, B, C, tile_size


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [_gen_case(rng) for _ in range(10)]
    # force an exact tile-size divisor case and a remainder case explicitly
    cases.append((rng.standard_normal(12), rng.standard_normal(12), rng.standard_normal(12), 4))
    cases.append((rng.standard_normal(10), rng.standard_normal(10), rng.standard_normal(10), 3))

    worst = 0.0
    for A, B, C, tile_size in cases:
        E_ref, F_ref = _baseline(A, B, C)
        try:
            E_got, F_got = sol.fused_tile_pipeline(A.copy(), B.copy(), C.copy(), tile_size)
            E_got = np.asarray(E_got, dtype=np.float64)
            if E_got.shape != E_ref.shape:
                worst = float("inf")
                break
            e1 = scorers.max_abs_err(E_ref, E_got)
            e2 = abs(float(F_got) - F_ref)
            err = max(e1, e2)
        except Exception:
            worst = float("inf")
            break
        worst = max(worst, err)
    return {"max_abs_err": worst}
