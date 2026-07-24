import numpy as np


def best_contiguous_n_block_to_drop(hidden_states: np.ndarray, n: int) -> tuple[int, float]:
    H = np.asarray(hidden_states, dtype=np.float64)
    _, L, _ = H.shape

    scores = []
    for s in range(L - n):
        a = H[:, s, :]
        b = H[:, s + n, :]
        cosine = np.sum(a * b, axis=1)
        cosine /= np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cosine = np.clip(cosine, -1.0, 1.0)
        scores.append(float(np.mean(np.arccos(cosine))))

    idx = int(np.argmin(np.asarray(scores)))
    return idx, scores[idx]
