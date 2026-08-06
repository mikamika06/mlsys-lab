import math
import numpy as np


def _quantize_rows(x):
    rows, cols = x.shape
    q = np.empty_like(x)
    scale_col = np.empty((rows, 1), dtype=x.dtype)
    for i in range(rows):
        max_val = 0.0
        for j in range(cols):
            val = x[i, j]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
        scale = max_val / 127.0
        scale_col[i, 0] = scale
        if scale == 0.0:
            for j in range(cols):
                q[i, j] = 0.0
        else:
            for j in range(cols):
                val = x[i, j] / scale
                if val >= 0.0:
                    fractional = val - math.floor(val)
                    if fractional == 0.5:
                        if math.floor(val) % 2 == 0:
                            q[i, j] = math.floor(val)
                        else:
                            q[i, j] = math.ceil(val)
                    elif fractional > 0.5:
                        q[i, j] = math.ceil(val)
                    else:
                        q[i, j] = math.floor(val)
                else:
                    pos_val = -val
                    fractional = pos_val - math.floor(pos_val)
                    if fractional == 0.5:
                        if math.floor(pos_val) % 2 == 0:
                            res = math.floor(pos_val)
                        else:
                            res = math.ceil(pos_val)
                        q[i, j] = -res
                    elif fractional > 0.5:
                        q[i, j] = -math.ceil(pos_val)
                    else:
                        q[i, j] = -math.floor(pos_val)
    return q * scale_col


def search_awq_alpha(W, X, s_x):
    alphas = np.arange(20, dtype=np.float64) / 20.0
    
    w_rows, w_cols = W.shape
    x_rows, x_cols = X.shape
    
    target = np.empty((w_rows, x_cols), dtype=W.dtype)
    for i in range(w_rows):
        for j in range(x_cols):
            acc = 0.0
            for k in range(w_cols):
                acc += W[i, k] * X[k, j]
            target[i, j] = acc

    losses = []

    for alpha_idx in range(len(alphas)):
        alpha = alphas[alpha_idx]
        
        s = np.empty_like(s_x)
        for i in range(len(s_x)):
            s[i] = math.pow(s_x[i], alpha)
            
        scaled = np.empty_like(W)
        for i in range(w_rows):
            for j in range(w_cols):
                scaled[i, j] = W[i, j] * s[j]
                
        quantized = _quantize_rows(scaled)
        
        restored = np.empty_like(quantized)
        for i in range(w_rows):
            for j in range(w_cols):
                inv_s = 1.0 / s[j]
                restored[i, j] = quantized[i, j] * inv_s
                
        restored_X = np.empty((w_rows, x_cols), dtype=W.dtype)
        for i in range(w_rows):
            for j in range(x_cols):
                acc = 0.0
                for k in range(w_cols):
                    acc += restored[i, k] * X[k, j]
                restored_X[i, j] = acc
                
        diff_sq_sum = 0.0
        for i in range(w_rows):
            for j in range(x_cols):
                diff = target[i, j] - restored_X[i, j]
                diff_sq_sum += diff * diff
        loss = math.sqrt(diff_sq_sum)
        losses.append(loss)

    losses_arr = np.asarray(losses, dtype=np.float64)
    
    best_idx = 0
    min_loss = losses_arr[0]
    for i in range(1, len(losses_arr)):
        if losses_arr[i] < min_loss:
            min_loss = losses_arr[i]
            best_idx = i
            
    return int(best_idx), losses_arr
