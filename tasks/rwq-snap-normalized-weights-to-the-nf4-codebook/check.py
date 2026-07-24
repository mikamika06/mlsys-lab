import numpy as np

def _nf4_levels() -> np.ndarray:
    return np.array(
        [
            -1.0,
            -0.93333333,
            -0.8,
            -0.66666667,
            -0.53333333,
            -0.4,
            -0.26666667,
            -0.13333333,
            0.0,
            0.13333333,
            0.26666667,
            0.4,
            0.53333333,
            0.66666667,
            0.8,
            0.93333333,
        ],
        dtype=np.float64,
    )


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    tests = [
        rng.uniform(-1, 1, size=100),
        rng.uniform(-1, 1, size=50),
        rng.standard_normal(200),   # may produce out‑of‑range values; clip to [-1,1]
        rng.integers(-10, 11, size=30) / 10.0,
    ]
    levels = _nf4_levels()
    ok = 1.0
    for w in tests:
        w_clipped = np.clip(w, -1.0, 1.0)
        try:
            got = sol.snap_nf4(np.asarray(w_clipped))
        except Exception:
            return {"exact_match": 0.0}
        ref = (
            np.argmin(
                np.abs(w_clipped[:, None] - levels[None, :]), axis=1
            )
            .astype(np.uint8)
        )
        if not np.array_equal(got, ref):
            ok = 0.0
            break
    return {"exact_match": ok}
