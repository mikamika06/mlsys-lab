from partitioner.predict import partition_ops

def optimize_allowlist(ops, base_allowlist, candidates, op_sizes, blob_overhead):
    """
    Evaluate candidate op types to add to the base_allowlist.
    Returns a tuple (best_candidate, best_total_bytes).
    If no candidate improves the total byte size compared to the base allowlist,
    return (None, base_total_bytes).
    """
    raise NotImplementedError
