def prune_unstructured(w, scores, sparsity):
    """
    Prunes the smallest scores in each row to reach the target sparsity.
    Returns: (w_pruned, boolean_mask)
    """
    raise NotImplementedError


def correct_bias(w, w_pruned, x):
    """
    Computes a bias vector to correct the mean shift caused by pruning.
    Returns: bias array of shape (out_features,)
    """
    raise NotImplementedError
