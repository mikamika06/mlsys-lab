import numpy as np

def min_diversity_bound(teacher_logits, rank, target_mse):
    V = teacher_logits.shape[0]
    norms = np.sum(teacher_logits ** 2, axis=1)
    sorted_indices = np.argsort(-norms)

    for K in range(1, V + 1):
        visited = sorted_indices[:K]
        T_sub = teacher_logits[visited]

        U, S, Vh = np.linalg.svd(T_sub, full_matrices=False)
        r = min(rank, len(S))

        truncation_error = np.sum(S[r:] ** 2)
        unvisited_error = np.sum(norms[sorted_indices[K:]])

        total_mse = (truncation_error + unvisited_error) / V
        if total_mse <= target_mse:
            return K

    return V
