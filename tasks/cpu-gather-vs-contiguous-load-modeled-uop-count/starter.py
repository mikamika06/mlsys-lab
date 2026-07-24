def modeled_load_uops(m, vw, e):
    """Return dict with modeled load-uop counts and simulated cache miss counts.

    Keys: contiguous_uops, gather_uops, contiguous_misses, gather_misses.
    Cache: line_bytes=64, sets=64, ways=8.  Element i → byte address i*e.
    Use random.seed(42) + shuffle for the gather access order.
    """
    raise NotImplementedError("your code here")
