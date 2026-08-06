import math
import numpy as np

def decompose_sdpa(Q, K, V, mask=None, scale=None):
    d_k = Q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)

    q_shape = Q.shape
    v_shape = V.shape
    
    q_ndim = Q.ndim
    k_ndim = K.ndim
    v_ndim = V.ndim

    if q_ndim == 2:
        q_batch = ()
        seq_q = q_shape[0]
    else:
        q_batch = q_shape[:-2]
        seq_q = q_shape[-2]

    if k_ndim == 2:
        seq_k = K.shape[0]
        dim_k = K.shape[1]
    else:
        seq_k = K.shape[-2]
        dim_k = K.shape[-1]

    dim_v = V.shape[-1]

    total_batches = 1
    for b in q_batch:
        total_batches *= b

    output_shape = q_batch + (seq_q, dim_v)
    output_flat = np.zeros(total_batches * seq_q * dim_v, dtype=Q.dtype)
    output = output_flat.reshape(output_shape)

    Q_flat = Q.reshape(total_batches, seq_q, d_k) if q_ndim > 2 else Q.reshape(1, seq_q, d_k)
    
    if k_ndim > 2:
        K_flat = K.reshape(total_batches, seq_k, dim_k)
    else:
        K_flat = K.reshape(1, seq_k, dim_k)

    if v_ndim > 2:
        V_flat = V.reshape(total_batches, seq_k, dim_v)
    else:
        V_flat = V.reshape(1, seq_k, dim_v)

    if mask is not None:
        mask_shape = mask.shape
        if len(mask_shape) == 2:
            mask_flat = mask
        else:
            mask_flat = mask

    out_flat_view = output.reshape(total_batches, seq_q, dim_v)

    for b in range(total_batches):
        q_b = Q_flat[b]
        k_b = K_flat[b]
        v_b = V_flat[b]

        scores = np.zeros((seq_q, seq_k), dtype=Q.dtype)
        for i in range(seq_q):
            for j in range(seq_k):
                acc = 0.0
                for d in range(d_k):
                    acc += q_b[i, d] * k_b[j, d]
                scores[i, j] = acc * scale

        if mask is not None:
            if mask.ndim == 2:
                m_b = mask_flat
            else:
                m_b = mask_flat[b]
            for i in range(seq_q):
                for j in range(seq_k):
                    scores[i, j] += m_b[i, j]

        weights = np.zeros((seq_q, seq_k), dtype=Q.dtype)
        for i in range(seq_q):
            max_val = -float('inf')
            for j in range(seq_k):
                if scores[i, j] > max_val:
                    max_val = scores[i, j]
            
            sum_exp = 0.0
            row_exp = np.zeros(seq_k, dtype=Q.dtype)
            for j in range(seq_k):
                val = math.exp(scores[i, j] - max_val)
                row_exp[j] = val
                sum_exp += val

            for j in range(seq_k):
                weights[i, j] = row_exp[j] / sum_exp

        out_b = np.zeros((seq_q, dim_v), dtype=Q.dtype)
        for i in range(seq_q):
            for c in range(dim_v):
                acc = 0.0
                for j in range(seq_k):
                    acc += weights[i, j] * v_b[j, c]
                out_b[i, c] = acc

        out_flat_view[b] = out_b

    return output
