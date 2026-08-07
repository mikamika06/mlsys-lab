def prefill_flops_saved_fraction(lengths: list[float], reused_prefix: list[float]) -> float:
    """
    lengths: (n,) full context length L_i of each request in the batch.
    reused_prefix: (n,) number of leading tokens of request i whose KV is
    already cached from a prior request sharing that prefix
    (0 <= reused_prefix_i <= lengths_i).

    Return the fraction of total batch prefill FLOPs saved by reusing
    those cached prefixes, under a causal prefill cost model where
    position i costs FLOPs proportional to i.
    """
    raise NotImplementedError('your code here')
