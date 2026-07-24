from mlsys.scorers import rel_err

def _reference(m, n, k, elem_bytes=8):
    flops = 2 * m * n * k
    bytes_moved = (m * k + k * n + m * n) * elem_bytes
    return flops / bytes_moved

def grade(sol, fx) -> dict:
    cases = [
        (1, 1, 1),
        (10, 20, 5),
        (64, 128, 32),
        (256, 512, 128),
        (1000, 2000, 500)
    ]
    max_err = 0.0
    for m, n, k in cases:
        try:
            got = sol.arithmetic_intensity(m, n, k)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _reference(m, n, k)
        err = rel_err(ref, got)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
