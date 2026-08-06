def compute_memory_usage(nodes, fused_kernels, inplace_buffers=False):
    """
    Given graph nodes and planned fused kernels, calculate total peak memory in bytes.
    Each buffer size is prod(shape) * dtype_size (assume float32 = 4 bytes).
    """
    raise NotImplementedError
