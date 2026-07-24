import numpy as np


def _softmax_entropy(scores):
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs = probs / np.sum(probs)
    return probs, -np.sum(probs * np.log(probs + 1e-12))


def compare_sink_window(Q, K, V, sink_size, window_size):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    T, d = Q.shape
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

        sink_scores = (Q[t] @ K[sink_indices].T) / np.sqrt(float(d))
        sink_probs, sink_h = _softmax_entropy(sink_scores)

        pure_scores = (Q[t] @ K[pure_indices].T) / np.sqrt(float(d))
        pure_probs, pure_h = _softmax_entropy(pure_scores)

        sink_outputs.append(sink_probs @ V[sink_indices])
        pure_outputs.append(pure_probs @ V[pure_indices])
        sink_entropy.append(sink_h)
        pure_entropy.append(pure_h)

    return {
        "sink_outputs": np.asarray(sink_outputs, dtype=np.float64),
        "pure_outputs": np.asarray(pure_outputs, dtype=np.float64),
        "sink_entropy": np.asarray(sink_entropy, dtype=np.float64),
        "pure_entropy": np.asarray(pure_entropy, dtype=np.float64),
    }
