def benchmark_mps_vs_eager(
    graph_fn, eager_fn, inputs: tuple, warmup: int = 5, runs: int = 20
) -> dict:
    """Measures latency and speedup ratio of graph execution vs eager execution."""
    raise NotImplementedError
