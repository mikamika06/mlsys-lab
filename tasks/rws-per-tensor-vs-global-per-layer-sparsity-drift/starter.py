def global_threshold_layer_sparsity(weights: list[list[float]], prune_ratio: float) -> dict:
    """Pool |w| across ALL layers, k=round(prune_ratio*N), prune the k
    globally-smallest-magnitude elements (stable ascending sort, ties to
    lower index). Return {"sparsity": (L,) float64 fraction pruned per
    layer under that shared threshold, "most_pruned_layer": int argmax}."""
    raise NotImplementedError('your code here')
