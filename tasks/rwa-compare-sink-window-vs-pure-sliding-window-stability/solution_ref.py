import math
import numpy as np


def _softmax_entropy(scores):
    max_score = scores[0]
    for s in scores:
        if s > max_score:
            max_score = s

    shifted = [s - max_score for s in scores]
    probs_list = [math.exp(s) for s in shifted]

    sum_probs = 0.0
    for p in probs_list:
        sum_probs += p

    probs_list = [p / sum_probs for p in probs_list]

    entropy = 0.0
    for p in probs_list:
        entropy += p * math.log(p + 1e-12)
    entropy = -entropy

    return np.array(probs_list, dtype=np.float64), entropy


def compare_sink_window(Q, K, V, sink_size, window_size):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    T, d = Q.shape
    scale = math.sqrt(float(d))

    sink_outputs = []
    pure_outputs = []
    sink_entropy = []
    pure_entropy = []

    for t in range(T):
        recent_start = max(0, t + 1 - window_size)
        sink_indices = list(range(min(sink_size, t + 1)))
        for i in range(recent_start, t + 1):
            if i not in sink_indices:
                sink_indices.append(i)

        pure_indices = list(range(recent_start, t + 1))

        Qt = Q[t]

        sink_scores = []
        for idx in sink_indices:
            K_row = K[idx]
            dot = 0.0
            for k in range(d):
                dot += Qt[k] * K_row[k]
            sink_scores.append(dot / scale)

        pure_scores = []
        for idx in pure_indices:
            K_row = K[idx]
            dot = 0.0
            for k in range(d):
                dot += Qt[k] * K_row[k]
            pure_scores.append(dot / scale)

        sink_probs, sink_h = _softmax_entropy(sink_scores)
        pure_probs, pure_h = _softmax_entropy(pure_scores)

        v_dim = V.shape[1]
        sink_out = [0.0] * v_dim
        for p_idx, p_val in enumerate(sink_probs):
            v_row = V[sink_indices[p_idx]]
            for v_col in range(v_dim):
                sink_out[v_col] += p_val * v_row[v_col]

        pure_out = [0.0] * v_dim
        for p_idx, p_val in enumerate(pure_probs):
            v_row = V[pure_indices[p_idx]]
            for v_col in range(v_dim):
                pure_out[v_col] += p_val * v_row[v_col]

        sink_outputs.append(sink_out)
        pure_outputs.append(pure_out)
        sink_entropy.append(sink_h)
        pure_entropy.append(pure_h)

    return {
        "sink_outputs": np.asarray(sink_outputs, dtype=np.float64),
        "pure_outputs": np.asarray(pure_outputs, dtype=np.float64),
        "sink_entropy": np.asarray(sink_entropy, dtype=np.float64),
        "pure_entropy": np.asarray(pure_entropy, dtype=np.float64),
    }
