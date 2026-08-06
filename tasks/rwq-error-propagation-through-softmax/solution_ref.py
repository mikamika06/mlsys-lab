import math
import numpy as np

def kv_quant_error_propagation(Q, K, V, K_hat, V_hat, scale=None):
    """Compute how KV quantization error propagates through softmax attention.

    Returns dict with keys: output_mse, kv_error, amplification.
    """
    d = Q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    def _attn(Q_, K_, V_):
        seq_q = Q_.shape[0]
        seq_k = K_.shape[0]
        d_v = V_.shape[1]

        logits = np.zeros((seq_q, seq_k), dtype=Q_.dtype)
        for i in range(seq_q):
            for j in range(seq_k):
                dot_val = 0.0
                for k_idx in range(d):
                    dot_val += Q_[i, k_idx] * K_[j, k_idx]
                logits[i, j] = dot_val * scale

        logits_max = np.zeros((seq_q, 1), dtype=Q_.dtype)
        for i in range(seq_q):
            m_val = logits[i, 0]
            for j in range(1, seq_k):
                if logits[i, j] > m_val:
                    m_val = logits[i, j]
            logits_max[i, 0] = m_val

        exp_logits = np.zeros((seq_q, seq_k), dtype=Q_.dtype)
        for i in range(seq_q):
            for j in range(seq_k):
                exp_logits[i, j] = math.exp(logits[i, j] - logits_max[i, 0])

        sum_exp = np.zeros((seq_q, 1), dtype=Q_.dtype)
        for i in range(seq_q):
            s_val = 0.0
            for j in range(seq_k):
                s_val += exp_logits[i, j]
            sum_exp[i, 0] = s_val

        weights = np.zeros((seq_q, seq_k), dtype=Q_.dtype)
        for i in range(seq_q):
            for j in range(seq_k):
                weights[i, j] = exp_logits[i, j] / sum_exp[i, 0]

        out = np.zeros((seq_q, d_v), dtype=Q_.dtype)
        for i in range(seq_q):
            for j in range(d_v):
                acc = 0.0
                for k_idx in range(seq_k):
                    acc += weights[i, k_idx] * V_[k_idx, j]
                out[i, j] = acc

        return weights, out

    _, O = _attn(Q, K, V)
    _, O_hat = _attn(Q, K_hat, V_hat)

    sq_diff_sum_o = 0.0
    count_o = 0
    for i in range(O.shape[0]):
        for j in range(O.shape[1]):
            diff = O[i, j] - O_hat[i, j]
            sq_diff_sum_o += diff * diff
            count_o += 1
    output_mse = float(sq_diff_sum_o / count_o)

    sq_diff_sum_k = 0.0
    count_k = 0
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            diff = K[i, j] - K_hat[i, j]
            sq_diff_sum_k += diff * diff
            count_k += 1
    mean_k_err = sq_diff_sum_k / count_k

    sq_diff_sum_v = 0.0
    count_v = 0
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            diff = V[i, j] - V_hat[i, j]
            sq_diff_sum_v += diff * diff
            count_v += 1
    mean_v_err = sq_diff_sum_v / count_v

    kv_error = float((mean_k_err + mean_v_err) / 2.0)
    amplification = output_mse / kv_error if kv_error > 0 else 0.0

    return {
        "output_mse": output_mse,
        "kv_error": kv_error,
        "amplification": amplification,
    }
