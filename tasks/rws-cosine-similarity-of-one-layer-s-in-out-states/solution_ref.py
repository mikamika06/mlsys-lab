import numpy as np

def mean_cosine_similarity(in_states: np.ndarray,
                           out_states: np.ndarray) -> float:
    in_norms = np.linalg.norm(in_states, axis=1)
    out_norms = np.linalg.norm(out_states, axis=1)
    cos = np.sum(in_states * out_states, axis=1) / (in_norms * out_norms)
    return float(np.mean(cos))
