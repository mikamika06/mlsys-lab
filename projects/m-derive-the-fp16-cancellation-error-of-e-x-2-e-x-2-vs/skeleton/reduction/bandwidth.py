def softmax_memory_traffic(N, element_bytes, fused=False):
    """
    Returns the total bytes read and written for a softmax operation on a vector of size N.
    Assume 6N elements of traffic for unfused, and 2N for fused.
    """
    raise NotImplementedError

def achieved_bandwidth_GBps(N, element_bytes, time_ms, fused=False):
    """
    Returns the achieved memory bandwidth in GB/s given the execution time in milliseconds.
    """
    raise NotImplementedError
