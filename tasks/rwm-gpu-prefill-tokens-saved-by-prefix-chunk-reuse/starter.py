def prefix_reuse_savings(trace: list[list[int]], chunk_size: int=512) -> int:
    """Broken implementation – only caches whole requests.
This fails to reuse any prefixes unless the entire request has been seen before."""
    raise NotImplementedError('your code here')
