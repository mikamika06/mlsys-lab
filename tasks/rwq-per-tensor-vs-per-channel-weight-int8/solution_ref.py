import numpy as np


def _sym_int8_quant(g: np.ndarray) -> np.ndarray:
    amax = float(np.max(np.abs(g)))
    scale = amax / 127.0 if amax > 0 else 1.0
    codes = np.clip(np.round(g / scale), -127, 127)
    return codes * scale


def int8_mse_per_tensor_vs_per_channel(W: np.ndarray):
    """Compare symmetric int8 reconstruction MSE: one tensor-wide scale
    vs one scale per output row (per channel).

    Returns (mse_per_tensor, mse_per_channel).
    """
    W = np.asarray(W, dtype=np.float64)

    W_hat_pt = _sym_int8_quant(W.reshape(-1)).reshape(W.shape)
    mse_per_tensor = float(np.mean((W_hat_pt - W) ** 2))

    W_hat_pc = np.empty_like(W)
    for i in range(W.shape[0]):
        W_hat_pc[i] = _sym_int8_quant(W[i])
    mse_per_channel = float(np.mean((W_hat_pc - W) ** 2))

    return mse_per_tensor, mse_per_channel
