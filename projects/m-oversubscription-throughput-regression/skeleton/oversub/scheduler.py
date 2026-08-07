def sweep_stream_count(bench_fn, max_streams):
    """Profile stream count performance and find the optimal knee point."""
    raise NotImplementedError


class OptimalStreamPool:
    """Stream pool that restricts concurrency to the optimal knee point."""

    def __init__(self, bench_fn, max_streams):
        raise NotImplementedError

    def get_optimal_streams(self):
        raise NotImplementedError

    def compute_throughput_ratio(self, oversubscribed_streams):
        raise NotImplementedError
