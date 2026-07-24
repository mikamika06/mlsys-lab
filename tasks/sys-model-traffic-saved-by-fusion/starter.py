def fusion_traffic(n: int, k: int, dtype_bytes: int):
    """
    Return (bytes_unfused, bytes_fused): total HBM read+write traffic for
    running a chain of k elementwise ops over n elements as k separate
    (unfused) kernel launches vs as one fused kernel.

    bytes_unfused = 2 * k * n * dtype_bytes  (every intermediate is read
      back from HBM by the next op, and written to HBM by the previous one)
    bytes_fused = 2 * n * dtype_bytes  (only the original input is read and
      the final output is written; intermediates never touch HBM)
    """
    raise NotImplementedError('your code here')
