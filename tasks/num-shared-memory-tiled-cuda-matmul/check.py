import numpy as np


def _tiled_load_oracle(m, k, n, tile_size):
    loads = 0
    T = int(tile_size)
    for row0 in range(0, m, T):
        for col0 in range(0, n, T):
            for t0 in range(0, k, T):
                rows = min(T, m - row0)
                cols = min(T, n - col0)
                depth = min(T, k - t0)
                loads += rows * depth + depth * cols
    return loads


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
            2,
        ),
        (
            np.arange(15, dtype=np.float64).reshape(3, 5) / 7.0,
            np.arange(20, dtype=np.float64).reshape(5, 4) / 11.0,
            2,
        ),
        (
            np.random.default_rng(4).normal(size=(7, 9)),
            np.random.default_rng(5).normal(size=(9, 6)),
            4,
        ),
    ]

    max_error = 0.0
    access_ratios = []

    for A, B, tile in cases:
        try:
            C, loads = sol.tiled_cuda_matmul(A, B, tile)
        except Exception:
            return {
                "max_abs_err": float("inf"),
                "modeled_access_count": float("inf"),
            }

        ref = np.matmul(A, B).astype(np.float64)
        max_error = max(
            max_error,
            float(np.max(np.abs(np.asarray(C, dtype=np.float64) - ref))),
        )

        oracle_loads = _tiled_load_oracle(
            A.shape[0], A.shape[1], B.shape[1], tile
        )
        access_ratios.append(float(loads) / float(oracle_loads))

    return {
        "max_abs_err": max_error,
        "modeled_access_count": max(access_ratios),
    }
