import numpy as np


def quantize_per_tensor_int8(w: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(w))
    scale = max_val / 127.0 if max_val > 0 else 1.0
    q = np.clip(np.round(w / scale), -128, 127)
    return q * scale


def quantize_per_row_int8(w: np.ndarray) -> np.ndarray:
    max_vals = np.max(np.abs(w), axis=1, keepdims=True)
    scales = np.where(max_vals > 0, max_vals / 127.0, 1.0)
    q = np.clip(np.round(w / scales), -128, 127)
    return q * scales


def compute_mse(w_orig: np.ndarray, w_quant: np.ndarray) -> float:
    diff = w_orig.astype(np.float64) - w_quant.astype(np.float64)
    return float(np.mean(diff ** 2))


def compare_error_metrics(w: np.ndarray) -> dict[str, float]:
    q_pt = quantize_per_tensor_int8(w)
    q_pr = quantize_per_row_int8(w)
    return {
        "per_tensor_mse": compute_mse(w, q_pt),
        "per_row_mse": compute_mse(w, q_pr),
    }
