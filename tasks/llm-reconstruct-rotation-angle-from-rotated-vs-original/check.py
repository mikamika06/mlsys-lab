import numpy as np
from mlsys.scorers import max_abs_err

def _reference(orig: np.ndarray, rot: np.ndarray) -> np.ndarray:
    cross = orig[:, 0] * rot[:, 1] - orig[:, 1] * rot[:, 0]
    dot   = orig[:, 0] * rot[:, 0] + orig[:, 1] * rot[:, 1]
    return np.arctan2(cross, dot)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = []
    for _ in range(5):
        n = rng.integers(10, 50)
        orig = rng.standard_normal((n, 2))
        angles = rng.uniform(-np.pi, np.pi, size=n)
        rot_mat = np.array([[np.cos(angles), -np.sin(angles)],
                            [np.sin(angles),  np.cos(angles)]])
        rot = orig @ rot_mat.T
        cases.append((orig, rot))

    ok = 1.0
    for orig, rot in cases:
        try:
            got = sol.recover_angles(orig, rot)
        except Exception:
            return {"max_abs_err": 0.0}
        ref = _reference(orig, rot)
        err = max_abs_err(ref, got)
        if err > 1e-5:
            ok = 0.0
            break
    return {"max_abs_err": ok}
