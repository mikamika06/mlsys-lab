def route_allocation(nbytes: int):
    """
    Route a raw allocation request through the size-rounding + pool-selection
    logic of a two-pool caching allocator.

    Returns (pool, segment_size):
      pool         : "small" if the ROUNDED request size is <= 1 MiB,
                     otherwise "large".
      segment_size : the size (in bytes) of the memory segment that would be
                      carved out / reserved to service this request.
    """
    raise NotImplementedError('your code here')
