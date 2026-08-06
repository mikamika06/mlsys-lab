import numpy as np

def min_diversity_bound(teacher_logits, rank, target_mse):
    V, d = teacher_logits.shape
    norms = np.sum(teacher_logits ** 2, axis=1)
    sorted_idx = np.argsort(-norms)

    for k in range(1, V + 1):
        visited = sorted_idx[:k]
        T_sub = teacher_logits[visited]

        U, S, Vh = np.linalg.svd(T_sub, full_matrices=False)
        r = min(rank, len(S))

        trunc_err = np.sum(S[r:] ** 2)
        unvis_err = np.sum(norms[sorted_idx[k:]])

        total_mse = (trunc_err + unvis_err) / (V * d)
        if total_mse <= target_mse:
            return k

    return V
