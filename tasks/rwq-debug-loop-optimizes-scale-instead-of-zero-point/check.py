import numpy as np


def _oracle(W, scale, bits, iters):
    qmax = (1 << bits) - 1
    z = 0

    def reconstruct(z_value):
        q = np.clip(np.rint(W / scale) + z_value, 0, qmax)
        return scale * (q - z_value)

    for _ in range(iters):
        best_z = z
        best_err = None
        for candidate in range(z - 2, z + 3):
            candidate_err = float(np.sum((W - reconstruct(candidate)) ** 2))
            if best_err is None or candidate_err < best_err:
                best_err = candidate_err
                best_z = candidate
        z = best_z

    return reconstruct(z), z


def grade(sol, fx) -> dict:
    cases = [
        (np.array([-1.0, -0.2, 0.1, 0.8, 1.7, 3.1]), 0.5, 3, 5),
        (np.array([-4.0, -1.5, 0.0, 2.0, 5.5]), 0.75, 4, 6),
        (np.array([0.2, 0.4, 0.9, 1.1, 1.8]), 0.25, 3, 4),
    ]

    worst = 0.0
    for W, scale, bits, iters in cases:
        try:
            got, _ = sol.optimize_zero_point(W.copy(), scale, bits, iters)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref, _ = _oracle(W, scale, bits, iters)
        worst = max(worst, float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref))))

    return {"max_abs_err": worst}
