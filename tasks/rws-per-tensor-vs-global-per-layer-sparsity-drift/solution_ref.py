import numpy as np


def global_threshold_layer_sparsity(weights, prune_ratio: float) -> dict:
    sizes = [w.size for w in weights]
    abs_pool = np.concatenate([np.abs(np.asarray(w, dtype=np.float64)).ravel() for w in weights])
    N = abs_pool.size
    k = int(round(prune_ratio * N))

    order = np.argsort(abs_pool, kind="stable")
    pruned = np.zeros(N, dtype=bool)
    pruned[order[:k]] = True

    offsets = np.cumsum([0] + sizes)
    sparsity = np.array(
        [pruned[offsets[i]:offsets[i + 1]].mean() for i in range(len(weights))],
        dtype=np.float64,
    )
    most_pruned = int(np.argmax(sparsity))

    return {"sparsity": sparsity, "most_pruned_layer": most_pruned}
