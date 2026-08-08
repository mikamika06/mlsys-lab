import numpy as np
from fp8.descale import quantize_and_descale


def find_optimal_scale(
    x: np.ndarray, candidates: list[float]
) -> tuple[float, float]:
    best_scale = float(candidates[0])
    best_mse = float("inf")
    x_f32 = np.asarray(x, dtype=np.float32)
    for s in candidates:
        _, recon = quantize_and_descale(x_f32, float(s))
        mse = float(np.mean((x_f32 - recon) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_scale = float(s)
    return best_scale, best_mse
