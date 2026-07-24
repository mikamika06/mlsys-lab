import numpy as np


def global_unstructured_masks(weights: list, amount: float) -> list:
    """Global (cross-layer) L1-unstructured pruning, as in
    torch.nn.utils.prune.global_unstructured with L1Unstructured.

    weights: list of weight arrays (possibly different shapes AND scales).
    amount: fraction of the TOTAL element count (summed across all weights)
      to prune -- a single magnitude threshold is chosen over the
      concatenation of every |w_ij| across ALL tensors, not per tensor.

    Returns a list of boolean masks, same shapes as `weights`, True = kept,
    False = pruned (the round(amount * total_elements) globally-smallest
    magnitude elements across every tensor combined).
    """
    raise NotImplementedError('your code here')
