import math
import numpy as np


def _e4m3(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        val = x[idx]
        ax = abs(val)
        if ax > 0:
            vals = ax if ax < 448.0 else 448.0
            exp_val = math.floor(math.log2(vals))
            exp = exp_val if exp_val > -6 else -6
            base = 2.0 ** exp
            mant = vals / base - 1.0
            mant_q = round(mant * 8.0) / 8.0
            bq = base * (1.0 + mant_q)
            vals_q = bq if bq < 448.0 else 448.0
            out[idx] = math.copysign(vals_q, val)
    return out


def _quant_dequant(x, per_head):
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    if per_head:
        axis = (1, 2)
        out_shape = tuple(1 if i in axis else shape[i] for i in range(len(shape)))
        scale = np.zeros(out_shape, dtype=np.float64)
        for idx in np.ndindex(shape):
            out_idx = tuple(0 if i in axis else idx[i] for i in range(len(shape)))
            val = abs(x[idx])
            if val > scale[out_idx]:
                scale[out_idx] = val
        scale = scale / 448.0
    else:
        max_val = 0.0
        for idx in np.ndindex(shape):
            val = abs(x[idx])
            if val > max_val:
                max_val = val
        scale = max_val / 448.0

    if isinstance(scale, np.ndarray):
        for idx in np.ndindex(scale.shape):
            if scale[idx] < 1e-12:
                scale[idx] = 1e-12
    else:
        if scale < 1e-12:
            scale = 1e-12

    x_scaled = np.zeros_like(x)
    for idx in np.ndindex(shape):
        if per_head:
            s_idx = tuple(0 if i in axis else idx[i] for i in range(len(shape)))
            x_scaled[idx] = x[idx] / scale[s_idx]
        else:
            x_scaled[idx] = x[idx] / scale

    e4m3_res = _e4m3(x_scaled)
    out = np.zeros_like(x)
    for idx in np.ndindex(shape):
        if per_head:
            s_idx = tuple(0 if i in axis else idx[i] for i in range(len(shape)))
            out[idx] = e4m3_res[idx] * scale[s_idx]
        else:
            out[idx] = e4m3_res[idx] * scale
    return out


def _attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = Q.shape[-1]
    
    K_swapped = np.swapaxes(K, 1, 2)
    q_shape = Q.shape
    k_swap_shape = K_swapped.shape
    
    batch_shape = q_shape[:-2]
    M = q_shape[-2]
    K_dim = q_shape[-1]
    N = k_swap_shape[-1]
    
    scores_shape = batch_shape + (M, N)
    scores = np.zeros(scores_shape, dtype=np.float64)
    
    for batch_idx in np.ndindex(batch_shape):
        for i in range(M):
            for j in range(N):
                acc = 0.0
                for k in range(K_dim):
                    acc += Q[batch_idx + (i, k)] * K_swapped[batch_idx + (k, j)]
                scores[batch_idx + (i, j)] = acc / math.sqrt(d)
                
    max_shape = scores_shape[:-1] + (1,)
    max_vals = np.full(max_shape, -float('inf'), dtype=np.float64)
    for idx in np.ndindex(scores_shape):
        prefix = idx[:-1]
        max_idx = prefix + (0,)
        if scores[idx] > max_vals[max_idx]:
            max_vals[max_idx] = scores[idx]
            
    for idx in np.ndindex(scores_shape):
        prefix = idx[:-1]
        max_idx = prefix + (0,)
        scores[idx] -= max_vals[max_idx]
        
    probs = np.zeros_like(scores)
    for idx in np.ndindex(scores_shape):
        probs[idx] = math.exp(scores[idx])
        
    sum_shape = scores_shape[:-1] + (1,)
    sums = np.zeros(sum_shape, dtype=np.float64)
    for idx in np.ndindex(scores_shape):
        prefix = idx[:-1]
        sum_idx = prefix + (0,)
        sums[sum_idx] += probs[idx]
        
    for idx in np.ndindex(scores_shape):
        prefix = idx[:-1]
        sum_idx = prefix + (0,)
        probs[idx] /= sums[sum_idx]
        
    probs_shape = probs.shape
    v_shape = V.shape
    v_N = v_shape[-1]
    
    out_shape = probs_shape[:-1] + (v_N,)
    out = np.zeros(out_shape, dtype=np.float64)
    
    for batch_idx in np.ndindex(batch_shape):
        for i in range(probs_shape[-2]):
            for j in range(v_N):
                acc = 0.0
                for k in range(probs_shape[-1]):
                    acc += probs[batch_idx + (i, k)] * V[batch_idx + (k, j)]
                out[batch_idx + (i, j)] = acc
    return out


def _per_head_delta(O_ref, O_quant):
    O_ref = np.asarray(O_ref, dtype=np.float64)
    O_quant = np.asarray(O_quant, dtype=np.float64)
    batch_size = O_ref.shape[0]
    num_list = []
    den_list = []
    for i in range(batch_size):
        diff_item = O_quant[i] - O_ref[i]
        ref_item = O_ref[i]
        sum_sq_diff = 0.0
        for idx in np.ndindex(diff_item.shape):
            val = diff_item[idx]
            sum_sq_diff += val * val
        num_i = math.sqrt(sum_sq_diff)
        
        sum_sq_ref = 0.0
        for idx in np.ndindex(ref_item.shape):
            val = ref_item[idx]
            sum_sq_ref += val * val
        den_i = math.sqrt(sum_sq_ref) + 1e-12
        
        num_list.append(num_i)
        den_list.append(den_i)
        
    total = 0.0
    for n_val, d_val in zip(num_list, den_list):
        total += n_val / d_val
    return float(total / batch_size)


def kv_scale_granularity_delta(Q, K, V):
    O_ref = _attention(Q, K, V)

    K_pt = _quant_dequant(K, per_head=False)
    V_pt = _quant_dequant(V, per_head=False)
    O_pt = _attention(Q, K_pt, V_pt)

    K_ph = _quant_dequant(K, per_head=True)
    V_ph = _quant_dequant(V, per_head=True)
    O_ph = _attention(Q, K_ph, V_ph)

    per_tensor_delta = _per_head_delta(O_ref, O_pt)
    per_head_delta = _per_head_delta(O_ref, O_ph)
    return per_tensor_delta, per_head_delta
