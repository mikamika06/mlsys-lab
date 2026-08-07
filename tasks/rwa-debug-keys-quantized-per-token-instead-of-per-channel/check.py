import numpy as np


def _oracle_quantize_per_channel(K, bits=4):
    K_arr = np.asarray(K, dtype=np.float64)
    levels = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(K_arr), axis=0, keepdims=True) / levels
    scale = np.where(scale == 0, 1.0, scale)
    q = np.round(K_arr / scale)
    return q * scale


def grade(sol, fx) -> dict:
    cases = [
        [
            [100.0, 0.2, 0.1, 0.3],
            [101.0, 0.1, 0.2, 0.4],
            [99.0, 0.3, 0.2, 0.2],
            [102.0, 0.2, 0.1, 0.5],
        ],
        [
            [50.0, 1.0, 0.5, 0.2, 0.1],
            [48.0, 0.8, 0.4, 0.3, 0.2],
            [52.0, 1.2, 0.6, 0.1, 0.3],
            [51.0, 0.9, 0.5, 0.2, 0.2],
            [49.0, 1.1, 0.4, 0.3, 0.1],
        ],
    ]

    errors = []
    for K in cases:
        try:
            got = np.asarray(sol.quantize_keys_per_channel(K, 4), dtype=np.float64)
        except Exception:
            return {"mse": float("inf")}

        ref = _oracle_quantize_per_channel(K, 4)
        errors.append(float(np.mean((got - ref) ** 2)))

    return {"mse": float(np.mean(errors))}
