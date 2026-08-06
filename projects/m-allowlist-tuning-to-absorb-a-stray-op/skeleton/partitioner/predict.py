def partition_ops(ops, allowlist):
    """
    Given a list of ops (dicts with 'id' and 'type') and an allowlist of types,
    return a list of blob assignments. Unsupported ops get -1. Contiguous supported
    ops should be grouped into incrementing blob IDs (0, 1, 2...).
    """
    raise NotImplementedError
