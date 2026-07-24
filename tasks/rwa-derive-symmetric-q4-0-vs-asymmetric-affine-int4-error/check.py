import numpy as np


def _oracle(block):
    x = np.asarray(block, dtype=np.float64)

    amax = np.max(np.abs(x))
    s_sym = 2 * amax / 15
    q_sym = np.clip(np.round(x / s_sym), -8, 7)
    x_sym = s_sym * q_sym
    err_sym = np.linalg.norm(x_sym - x) / (np.linalg.norm(x) + 1e-12)

    xmin = np.min(x)
    xmax = np.max(x)
    s_aff = (xmax - xmin) / 15
    z = np.round(-xmin / s_aff)
    q_aff = np.clip(np.round(x / s_aff + z), 0, 15)
    x_aff = s_aff * (q_aff - z)
    err_aff = np.linalg.norm(x_aff - x) / (np.linalg.norm(x) + 1e-12)

    winner = "q4_0" if err_sym <= err_aff else "affine_int4"
    return np.array([err_sym, err_aff, 0.0 if winner == "q4_0" else 1.0])


def grade(sol, fx) -> dict:
    cases = [
        np.array([-1.0, 0.15, 0.2, 0.4, 1.0, 3.5, 4.0]),
        np.array([-0.5, 0.1, 0.2, 0.3, 2.0, 5.0]),
        np.array([-2.0, -0.1, 0.0, 0.4, 0.6, 8.0]),
    ]

    worst = 0.0
    for x in cases:
        try:
            got = sol.compare_q4_errors(x)
            vec = np.array([
                float(got["q4_0_error"]),
                float(got["affine_int4_error"]),
                0.0 if got["winner"] == "q4_0" else 1.0,
            ])
        except Exception:
            return {"rel_err": 1.0}

        ref = _oracle(x)
        denom = np.linalg.norm(ref) + 1e-12
        worst = max(worst, float(np.linalg.norm(vec - ref) / denom))

    return {"rel_err": worst}
