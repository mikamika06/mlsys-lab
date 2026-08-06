import numpy as np

def quantize_layer_output_mse(W: np.ndarray,
                              X: np.ndarray,
                              group_size: int = 16) -> float:
    out, in_ = W.shape
    padded_in = ((in_ + group_size - 1) // group_size) * group_size
    num_groups = padded_in // group_size
    
    W_hat = np.zeros((out, in_), dtype=W.dtype)
    
    for i in range(out):
        for g in range(num_groups):
            max_val = 0.0
            start_col = g * group_size
            for k in range(group_size):
                col = start_col + k
                val = W[i, col] if col < in_ else 0.0
                abs_val = val if val >= 0.0 else -val
                if abs_val > max_val:
                    max_val = abs_val
            
            scale = max_val / 7.0
            if scale == 0.0:
                scale = 1.0
            
            for k in range(group_size):
                col = start_col + k
                if col < in_:
                    val = W[i, col]
                    scaled_val = val / scale
                    q_val = round(scaled_val)
                    if q_val < -8:
                        q_val = -8
                    elif q_val > 7:
                        q_val = 7
                    deq_val = q_val * scale
                    W_hat[i, col] = deq_val

    cols = X.shape[1]
    WX = np.zeros((out, cols), dtype=W.dtype)
    for i in range(out):
        for j in range(cols):
            acc = 0.0
            for k in range(in_):
                acc += W[i, k] * X[k, j]
            WX[i, j] = acc

    W_hat_X = np.zeros((out, cols), dtype=W.dtype)
    for i in range(out):
        for j in range(cols):
            acc = 0.0
            for k in range(in_):
                acc += W_hat[i, k] * X[k, j]
            W_hat_X[i, j] = acc

    total_sum = 0.0
    count = out * cols
    for i in range(out):
        for j in range(cols):
            diff = WX[i, j] - W_hat_X[i, j]
            total_sum += diff * diff

    mse_val = float(total_sum / count)
    return mse_val
