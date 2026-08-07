def count_recompiles(shapes, bucket_size=64):
    """Count recompilations under bucketing."""
    seen = set()
    recompiles = 0
    for s in shapes:
        b = ((s + bucket_size - 1) // bucket_size) * bucket_size
        if b not in seen:
            seen.add(b)
            recompiles += 1
    return recompiles
