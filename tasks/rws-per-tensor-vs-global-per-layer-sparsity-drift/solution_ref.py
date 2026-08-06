import numpy as np


def global_threshold_layer_sparsity(weights, prune_ratio: float) -> dict:
    sizes = [w.size for w in weights]
    abs_pool_list = []
    for w in weights:
        arr = np.asarray(w, dtype=np.float64).ravel()
        for val in arr:
            abs_pool_list.append(abs(val))
    abs_pool = np.array(abs_pool_list, dtype=np.float64)
    N = abs_pool.size
    k = int(round(prune_ratio * N))

    indexed_pool = [(abs_pool[i], i) for i in range(N)]
    sorted_indexed = sorted(indexed_pool, key=lambda x: x[0])
    order = np.array([item[1] for item in sorted_indexed], dtype=np.intp)

    pruned = np.zeros(N, dtype=bool)
    for i in range(k):
        pruned[order[i]] = True

    cum_sizes = [0]
    curr = 0
    for s in sizes:
        curr += s
        cum_sizes.append(curr)
    offsets = np.array(cum_sizes, dtype=np.intp)

    sparsity_list = []
    for i in range(len(weights)):
        start = offsets[i]
        end = offsets[i + 1]
        sub = pruned[start:end]
        total_true = 0
        count = 0
        for val in sub:
            if val:
                total_true += 1
            count += 1
        mean_val = float(total_true) / count if count > 0 else 0.0
        sparsity_list.append(mean_val)

    sparsity = np.array(sparsity_list, dtype=np.float64)

    most_pruned = 0
    if len(sparsity) > 0:
        max_val = sparsity[0]
        for i in range(1, len(sparsity)):
            if sparsity[i] > max_val:
                max_val = sparsity[i]
                most_pruned = i

    return {"sparsity": sparsity, "most_pruned_layer": most_pruned}
