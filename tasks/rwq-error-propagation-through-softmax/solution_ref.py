import math

def kv_quant_error_propagation(Q: list[list[float]], K: list[list[float]], V: list[list[float]], K_hat: list[list[float]], V_hat: list[list[float]], scale: float | None = None) -> dict[str, float]:
    """Compute how KV quantization error propagates through softmax attention.

    Returns dict with keys: output_mse, kv_error, amplification.
    """
    d = len(Q[0])
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    def _attn(Q_, K_, V_):
        seq_q = len(Q_)
        seq_k = len(K_)
        d_v = len(V_[0])

        logits = [[0.0] * seq_k for _ in range(seq_q)]
        for i in range(seq_q):
            for j in range(seq_k):
                dot_val = 0.0
                for k_idx in range(d):
                    dot_val += Q_[i][k_idx] * K_[j][k_idx]
                logits[i][j] = dot_val * scale

        logits_max = [[0.0] for _ in range(seq_q)]
        for i in range(seq_q):
            m_val = logits[i][0]
            for j in range(1, seq_k):
                if logits[i][j] > m_val:
                    m_val = logits[i][j]
            logits_max[i][0] = m_val

        exp_logits = [[0.0] * seq_k for _ in range(seq_q)]
        for i in range(seq_q):
            for j in range(seq_k):
                exp_logits[i][j] = math.exp(logits[i][j] - logits_max[i][0])

        sum_exp = [[0.0] for _ in range(seq_q)]
        for i in range(seq_q):
            s_val = 0.0
            for j in range(seq_k):
                s_val += exp_logits[i][j]
            sum_exp[i][0] = s_val

        weights = [[0.0] * seq_k for _ in range(seq_q)]
        for i in range(seq_q):
            for j in range(seq_k):
                weights[i][j] = exp_logits[i][j] / sum_exp[i][0]

        out = [[0.0] * d_v for _ in range(seq_q)]
        for i in range(seq_q):
            for j in range(d_v):
                acc = 0.0
                for k_idx in range(seq_k):
                    acc += weights[i][k_idx] * V_[k_idx][j]
                out[i][j] = acc

        return weights, out

    _, O = _attn(Q, K, V)
    _, O_hat = _attn(Q, K_hat, V_hat)

    sq_diff_sum_o = 0.0
    count_o = 0
    for i in range(len(O)):
        for j in range(len(O[0])):
            diff = O[i][j] - O_hat[i][j]
            sq_diff_sum_o += diff * diff
            count_o += 1
    output_mse = float(sq_diff_sum_o / count_o)

    sq_diff_sum_k = 0.0
    count_k = 0
    for i in range(len(K)):
        for j in range(len(K[0])):
            diff = K[i][j] - K_hat[i][j]
            sq_diff_sum_k += diff * diff
            count_k += 1
    mean_k_err = sq_diff_sum_k / count_k

    sq_diff_sum_v = 0.0
    count_v = 0
    for i in range(len(V)):
        for j in range(len(V[0])):
            diff = V[i][j] - V_hat[i][j]
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
