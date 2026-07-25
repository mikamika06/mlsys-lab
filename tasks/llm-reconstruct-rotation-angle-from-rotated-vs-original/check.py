import numpy as np

from mlsys.scorers import max_abs_err


def _rotate(v, ang):
    """Rotate each row of v by its own angle. Written out per-row so the fixture
    is unambiguous: a batched matmul against a stacked rotation matrix broadcasts
    in a way that silently does something else."""
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(v)
    out[:, 0] = c * v[:, 0] - s * v[:, 1]
    out[:, 1] = s * v[:, 0] + c * v[:, 1]
    return out


def _reference(orig, rot):
    cross = orig[:, 0] * rot[:, 1] - orig[:, 1] * rot[:, 0]
    dot = orig[:, 0] * rot[:, 0] + orig[:, 1] * rot[:, 1]
    return np.arctan2(cross, dot)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    worst = 0.0
    for _ in range(5):
        n = int(rng.integers(10, 50))
        orig = rng.standard_normal((n, 2))
        ang = rng.uniform(-np.pi + 1e-3, np.pi - 1e-3, size=n)
        rot = _rotate(orig, ang)
        try:
            got = np.asarray(sol.recover_angles(orig, rot), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != (n,):
            return {"max_abs_err": float("inf")}
        # the angle the reference recovers is the one that was applied
        ref = _reference(orig, rot)
        worst = max(worst, float(max_abs_err(ref, got)), float(max_abs_err(ang, ref)))
    return {"max_abs_err": worst}
