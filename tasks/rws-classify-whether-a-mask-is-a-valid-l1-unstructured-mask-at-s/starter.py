def is_valid_l1_mask(w: list[float], mask: list[bool], amount: float) -> bool:
    """
    Return True iff `mask` is exactly the L1-unstructured pruning mask
    for `w` at sparsity `amount`: the k = round(amount * w.size)
    smallest-|w| entries are pruned (False), everything else is kept
    (True). See task.md for the exact rounding rule.
    """
    raise NotImplementedError('your code here')
