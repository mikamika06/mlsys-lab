import numpy as np


def compare_q4k_q40_error(weights: np.ndarray) -> dict:
    w = weights.astype(np.float32)
    mx = np.max(w)
    mn = np.min(w)
    scale = (mx - mn) / 15.0 if mx != mn else 1.0
    q0_q = np.clip(np.round((w - mn) / (scale if scale > 0 else 1.0)), 0, 15)
    q0_recon = q0_q * scale + mn
    q0_mse = float(np.mean((w - q0_recon) ** 2))

    from q4k.quant import dequantize_q4_k, quantize_q4_k
    data = quantize_q4_k(weights)
    recon = dequantize_q4_k(data)
    q4k_mse = float(np.mean((w - recon) ** 2))

    return {"q4_0_mse": q0_mse, "q4_k_mse": q4k_mse}
