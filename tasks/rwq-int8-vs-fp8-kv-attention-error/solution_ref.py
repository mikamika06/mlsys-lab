import math
import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    out = np.zeros_like(x)
    if not shape:
        return np.array(math.exp(x) / math.exp(x), dtype=np.float64)
    prefix_shape = shape[:-1]
    last_dim = shape[-1]
    if len(prefix_shape) == 0:
        max_val = x[0]
        for i in range(1, last_dim):
            if x[i] > max_val:
                max_val = x[i]
        s = 0.0
        e_list = []
        for i in range(last_dim):
            val = math.exp(x[i] - max_val)
            e_list.append(val)
            s += val
        for i in range(last_dim):
            out[i] = e_list[i] / s
    else:
        for idx in np.ndindex(prefix_shape):
            max_val = x[idx + (0,)]
            for i in range(1, last_dim):
                val = x[idx + (i,)]
                if val > max_val:
                    max_val = val
            s = 0.0
            e_list = []
            for i in range(last_dim):
                val = math.exp(x[idx + (i,)] - max_val)
                e_list.append(val)
                s += val
            for i in range(last_dim):
                out[idx + (i,)] = e_list[i] / s
    return out


def _matmul(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    M, K_a = A.shape
    K_b, N = B.shape
    out = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K_a):
                s += A[i, k] * B[k, j]
            out[i, j] = s
    return out


def _attention(Q, K, V):
    Q_f = np.asarray(Q, dtype=np.float64)
    K_f = np.asarray(K, dtype=np.float64)
    V_f = np.asarray(V, dtype=np.float64)
    seq_k, d = K_f.shape
    K_T = np.zeros((d, seq_k), dtype=np.float64)
    for i in range(seq_k):
        for j in range(d):
            K_T[j, i] = K_f[i, j]
    scores = _matmul(Q_f, K_T)
    scale = math.sqrt(d)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            scores[i, j] /= scale
    weights = _softmax(scores)
    return _matmul(weights, V_f)


def _int8_quant(x):
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    out = np.zeros_like(x)
    max_abs = 0.0
    for idx in np.ndindex(shape):
        val = abs(x[idx])
        if val > max_abs:
            max_abs = val
    scale = max_abs / 127.0
    if scale == 0.0:
        return out
    for idx in np.ndindex(shape):
        out[idx] = round(x[idx] / scale) * scale
    return out


def _fp8_e4m3_quant(x):
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    out = np.zeros_like(x)
    for idx in np.ndindex(shape):
        val = x[idx]
        if val != 0.0:
            ax = abs(val)
            exp = math.floor(math.log2(ax))
            if exp < -6:
                exp = -6
            elif exp > 7:
                exp = 7
            scale = math.pow(2.0, exp - 3)
            rounded = round(ax / scale) * scale
            if rounded > 240.0:
                rounded = 240.0
            sign = math.copysign(1.0, val)
            out[idx] = sign * rounded
    return out


def kv_attention_quant_error(Q, K, V):
    ref = _attention(Q, K, V)
    int8_out = _attention(Q, _int8_quant(K), _int8_quant(V))
    fp8_out = _attention(Q, _fp8_e4m3_quant(K), _fp8_e4m3_quant(V))
    
    shape = ref.shape
    total_elements = 1
    for dim in shape:
        total_elements *= dim
        
    int8_sum_sq = 0.0
    fp8_sum_sq = 0.0
    
    for idx in np.ndindex(shape):
        diff_int8 = ref[idx] - int8_out[idx]
        int8_sum_sq += diff_int8 * diff_int8
        
        diff_fp8 = ref[idx] - fp8_out[idx]
        fp8_sum_sq += diff_fp8 * diff_fp8
        
    int8_mse = float(int8_sum_sq / total_elements)
    fp8_mse = float(fp8_sum_sq / total_elements)
    
    winner = "int8" if int8_mse <= fp8_mse else "fp8"
    return int8_mse, fp8_mse, winner
