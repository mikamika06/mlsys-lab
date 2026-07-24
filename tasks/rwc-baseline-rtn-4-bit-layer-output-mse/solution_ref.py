import numpy as np

def quantize_layer_output_mse(W: np.ndarray,
                              X: np.ndarray,
                              group_size: int = 16) -> float:
    out, in_ = W.shape
    padded_in = ((in_ + group_size - 1) // group_size) * group_size
    W_pad = np.pad(W, ((0, 0), (0, padded_in - in_)), mode="constant")
    groups = W_pad.reshape(out, padded_in // group_size, group_size)
    max_abs = np.max(np.abs(groups), axis=2, keepdims=True)
    scale = max_abs / 7.0
    scale[scale == 0] = 1.0
    q = np.round(groups / scale).astype(np.int8)
    q = np.clip(q, -8, 7)
    deq = q * scale
    W_hat = deq.reshape(out, padded_in)[:, :in_]
    mse_val = float(np.mean((W @ X - W_hat @ X) ** 2))
    return mse_val
