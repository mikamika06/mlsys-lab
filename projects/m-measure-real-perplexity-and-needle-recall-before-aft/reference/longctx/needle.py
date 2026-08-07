import numpy as np

def measure_recall_at_k(retrieval_scores, needle_indices, k):
    top_k_indices = np.argsort(retrieval_scores, axis=-1)[:, -k:]
    hits = 0
    for i, needle in enumerate(needle_indices):
        if needle in top_k_indices[i]:
            hits += 1
    return float(hits / len(needle_indices))
