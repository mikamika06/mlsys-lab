def _flatten(obj):
    """Recursively flattens nested lists/tuples."""
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten(item)
    else:
        yield obj


def global_threshold_layer_sparsity(weights: list[list[float]], prune_ratio: float) -> dict:
    sizes = []
    abs_pool_list = []
    for w in weights:
        flat = list(_flatten(w))
        sizes.append(len(flat))
        for val in flat:
            abs_pool_list.append(abs(float(val)))

    N = len(abs_pool_list)
    k = int(round(prune_ratio * N))

    indexed_pool = [(abs_pool_list[i], i) for i in range(N)]
    sorted_indexed = sorted(indexed_pool, key=lambda x: x[0])
    order = [item[1] for item in sorted_indexed]

    pruned = [False] * N
    for i in range(k):
        pruned[order[i]] = True

    cum_sizes = [0]
    curr = 0
    for s in sizes:
        curr += s
        cum_sizes.append(curr)
    offsets = cum_sizes

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

    most_pruned = 0
    if len(sparsity_list) > 0:
        max_val = sparsity_list[0]
        for i in range(1, len(sparsity_list)):
            if sparsity_list[i] > max_val:
                max_val = sparsity_list[i]
                most_pruned = i

    return {"sparsity": sparsity_list, "most_pruned_layer": most_pruned}
