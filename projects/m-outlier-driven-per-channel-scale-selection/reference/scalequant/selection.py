import numpy as np
from scalequant.quant import quantize_dequantize_channel


def select_outlier_scales(x: np.ndarray, num_bits: int = 8, n_candidates: int = 32) -> np.ndarray:
    """Select optimal per-channel scales by minimizing MSE."""
    c_num = x.shape[0]
    qmax = (1 << (num_bits - 1)) - 1
    best_scales = np.zeros((c_num, 1), dtype=np.float64)
    grid = np.linspace(0.3, 1.0, n_candidates)

    for i in range(c_num):
        row = x[i : i + 1, :]
        max_v = np.max(np.abs(row))
        if max_v == 0:
            best_scales[i, 0] = 1.0
            continue
        s_max = max_v / qmax
        best_mse = float("inf")
        best_s = s_max
        for g in grid:
            s_cand = np.array([[s_max * g]], dtype=np.float64)
            deq = quantize_dequantize_channel(row, s_cand, num_bits)
            mse = float(np.mean((row - deq) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_s = s_cand[0, 0]
        best_scales[i, 0] = best_s
    return best_scales


def evaluate_quantization_loss(x: np.ndarray, scales: np.ndarray, num_bits: int = 8) -> float:
    """Compute total mean squared error over all channels."""
    deq = quantize_dequantize_channel(x, scales, num_bits)
    return float(np.mean((x - deq) ** 2))
