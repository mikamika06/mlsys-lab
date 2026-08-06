import math
import numpy as np


def _score_kv_tile(q_block, k_block, v_block, q_start, k_start, tile_size):
    q_len = q_block.shape[0]
    k_len = k_block.shape[0]
    d = q_block.shape[1]
    scale = math.sqrt(float(d))
    
    s = np.zeros((q_len, k_len), dtype=np.float64)
    for i in range(q_len):
        for j in range(k_len):
            dot = 0.0
            for k in range(d):
                dot += q_block[i, k] * k_block[j, k]
            s[i, j] = dot / scale
            
    return s, v_block


def causal_flash_attention_forward(Q, K, V, tile_size=2):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    dv = V.shape[1]
    out = np.zeros((n, dv), dtype=np.float64)
    lse = np.zeros(n, dtype=np.float64)

    for q_start in range(0, n, tile_size):
        q_end = min(n, q_start + tile_size)
        q_block = Q[q_start:q_end]
        scores_parts = []
        values_parts = []

        for k_start in range(0, n, tile_size):
            if k_start > q_end - 1:
                break
            k_end = min(n, k_start + tile_size)
            s, v = _score_kv_tile(
                q_block,
                K[k_start:k_end],
                V[k_start:k_end],
                q_start,
                k_start,
                tile_size,
            )
            
            q_len = q_end - q_start
            k_len = k_end - k_start
            masked_s = np.zeros((q_len, k_len), dtype=np.float64)
            for i in range(q_len):
                row_idx = q_start + i
                for j in range(k_len):
                    col_idx = k_start + j
                    if col_idx <= row_idx:
                        masked_s[i, j] = s[i, j]
                    else:
                        masked_s[i, j] = -float('inf')
            scores_parts.append(masked_s)
            values_parts.append(v)

        total_k_len = sum(part.shape[1] for part in scores_parts)
        q_len = q_end - q_start
        scores = np.zeros((q_len, total_k_len), dtype=np.float64)
        curr_col = 0
        for part in scores_parts:
            p_cols = part.shape[1]
            for i in range(q_len):
                for j in range(p_cols):
                    scores[i, curr_col + j] = part[i, j]
            curr_col += p_cols

        total_v_rows = sum(part.shape[0] for part in values_parts)
        vals = np.zeros((total_v_rows, dv), dtype=np.float64)
        curr_row = 0
        for part in values_parts:
            p_rows = part.shape[0]
            for i in range(p_rows):
                for j in range(dv):
                    vals[curr_row + i, j] = part[i, j]
            curr_row += p_rows

        m = np.zeros(q_len, dtype=np.float64)
        for i in range(q_len):
            max_val = -float('inf')
            for j in range(total_k_len):
                if scores[i, j] > max_val:
                    max_val = scores[i, j]
            m[i] = max_val

        e = np.zeros((q_len, total_k_len), dtype=np.float64)
        for i in range(q_len):
            for j in range(total_k_len):
                e[i, j] = math.exp(scores[i, j] - m[i])

        sum_e = np.zeros(q_len, dtype=np.float64)
        for i in range(q_len):
            s_val = 0.0
            for j in range(total_k_len):
                s_val += e[i, j]
            sum_e[i] = s_val

        p = np.zeros((q_len, total_k_len), dtype=np.float64)
        for i in range(q_len):
            for j in range(total_k_len):
                p[i, j] = e[i, j] / sum_e[i]

        block_out = np.zeros((q_len, dv), dtype=np.float64)
        for i in range(q_len):
            for j in range(dv):
                acc = 0.0
                for k in range(total_k_len):
                    acc += p[i, k] * vals[k, j]
                block_out[i, j] = acc
        out[q_start:q_end] = block_out

        block_lse = np.zeros(q_len, dtype=np.float64)
        for i in range(q_len):
            block_lse[i] = math.log(sum_e[i]) + m[i]
        lse[q_start:q_end] = block_lse

    return out, lse
