import numpy as np


def _oracle_quantize_absmax(x):
    x = np.asarray(x, dtype=np.float32)
    max_abs = float(np.max(np.abs(x)))
    scale = max_abs / 127.0 if max_abs != 0.0 else 1.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, float(scale)


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    cases = [
        np.array([-1.0, 0.5, 4.0, -8.0], dtype=np.float32),
        np.array([0.01, -0.02, 0.03, 15.0, -0.04], dtype=np.float32),
        np.array([1.0, -2.0, 3.0, 100.0, -75.0, 0.5], dtype=np.float32),
        np.linspace(-5, 5, 257, dtype=np.float32),
    ]

    worst = 0.0
    for x in cases:
        q_ref, s_ref = _oracle_quantize_absmax(x)
        ref = q_ref.astype(np.float64) * s_ref
        try:
            q, s = sol.quantize_absmax(x)
            got = np.asarray(q, dtype=np.float64) * float(s)
        except Exception:
            return {"rel_err": float("inf")}
        worst = max(worst, _rel_err(got, ref))

    return {"rel_err": worst}
