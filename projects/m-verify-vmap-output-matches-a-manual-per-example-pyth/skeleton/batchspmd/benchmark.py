def benchmark_vmap_speedup(fn_single, fn_batched, x_batches, axis=0, timer=None):
    """Measures execution speedup of fn_batched over per_example_loop for a list of x_batches."""
    raise NotImplementedError
