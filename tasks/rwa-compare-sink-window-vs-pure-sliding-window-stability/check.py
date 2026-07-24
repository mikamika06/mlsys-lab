import numpy as np


def _softmax_entropy(scores):
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs = probs / np.sum(probs)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return probs, entropy


def _oracle(Q, K, V, sink_size, window_size):
    T, d = Q.shape
    sink_outputs = []
    pure_outputs = []
    sink_entropy = []
    pure_entropy = []

    for t in range(T):
        scale = np.sqrt(float(d))
        full_scores = (Q[t] @ K[: t + 1].T) / scale
        full_probs, _ = _softmax_entropy(full_scores)

        sink_indices = list(range(min(sink_size, t + 1)))
        recent_start = max(0, t + 1 - window_size)
        for i in range(recent_start, t + 1):
            if i not in sink_indices:
                sink_indices.append(i)

        pure_indices = list(range(recent_start, t + 1))

        sink_scores = (Q[t] @ K[sink_indices].T) / scale
        sink_probs, sink_h = _softmax_entropy(sink_scores)

        pure_scores = (Q[t] @ K[pure_indices].T) / scale
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


def _pack(x):
    return np.concatenate(
        [
            x["sink_outputs"].ravel(),
            x["pure_outputs"].ravel(),
            x["sink_entropy"].ravel(),
            x["pure_entropy"].ravel(),
        ]
    )


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        (18, 4, 3, 5),
        (32, 6, 2, 7),
        (45, 5, 3, 8),
    ]

    errors = []
    for T, d, sink_size, window_size in cases:
        Q = rng.normal(size=(T, d)).astype(np.float64)
        K = rng.normal(size=(T, d)).astype(np.float64)
        V = rng.normal(size=(T, d + 1)).astype(np.float64)

        ref = _oracle(Q, K, V, sink_size, window_size)
        try:
            got = sol.compare_sink_window(Q, K, V, sink_size, window_size)
            err = np.linalg.norm(_pack(got) - _pack(ref)) / (
                np.linalg.norm(_pack(ref)) + 1e-12
            )
        except Exception:
            err = float("inf")
        errors.append(err)

    return {"rel_err": float(max(errors) if errors else float("inf"))}
