import numpy as np


def _oracle_round_trip(grad, scale):
    scaled = np.asarray(grad, dtype=np.float32) * np.float32(scale)
    low = scaled.astype(np.float16)
    restored = low.astype(np.float32) / np.float32(scale)
    return restored.astype(np.float32)


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array(
                [1e-8, -2e-8, 3e-7, -5e-7, 7e-9],
                dtype=np.float32,
            ),
            65536.0,
        ),
        (
            np.array([2e-10, -4e-10, 8e-10, 1.6e-9], dtype=np.float32),
            1048576.0,
        ),
        (
            np.array([3e-7, -1e-7, 9e-8, -6e-8], dtype=np.float32),
            32768.0,
        ),
    ]

    worst = 0.0
    beats = 1.0

    for grad, scale in cases:
        try:
            got = sol.loss_scale_round_trip(grad.copy(), scale)
        except Exception:
            return {"rel_err": 1.0, "beats_unscaled": 0.0}

        ref = _oracle_round_trip(grad, scale)
        err = _rel_err(grad, got)
        worst = max(worst, err)

        plain = grad.astype(np.float16).astype(np.float32)
        if err >= _rel_err(grad, plain):
            beats = 0.0

        if not np.all(np.isfinite(got)):
            beats = 0.0

    return {
        "rel_err": worst,
        "beats_unscaled": beats,
    }
