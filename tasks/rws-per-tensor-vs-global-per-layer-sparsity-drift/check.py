import numpy as np


def _oracle(weights, prune_ratio):
    sizes = [w.size for w in weights]
    abs_pool = np.concatenate([np.abs(w).ravel() for w in weights])
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
    return sparsity, most_pruned


def grade(sol, fx) -> dict:
    weights = [fx["layer0"], fx["layer1"], fx["layer2"], fx["layer3"]]
    prune_ratio = 0.5
    ref_sparsity, ref_most = _oracle(weights, prune_ratio)

    try:
        got = sol.global_threshold_layer_sparsity(
            [w.copy() for w in weights], prune_ratio
        )
        got_sparsity = np.asarray(got["sparsity"], dtype=np.float64)
        got_most = int(got["most_pruned_layer"])
    except Exception:
        return {"sparsity_exact": 0.0, "most_pruned_layer_exact": 0.0}

    if got_sparsity.shape != ref_sparsity.shape:
        return {"sparsity_exact": 0.0, "most_pruned_layer_exact": 0.0}

    sparsity_exact = 1.0 if np.array_equal(got_sparsity, ref_sparsity) else 0.0
    most_exact = 1.0 if got_most == ref_most else 0.0

    return {"sparsity_exact": sparsity_exact, "most_pruned_layer_exact": most_exact}
