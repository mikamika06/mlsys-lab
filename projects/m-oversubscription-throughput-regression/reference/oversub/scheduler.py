def sweep_stream_count(bench_fn, max_streams):
    """Profile stream count performance and find the optimal knee point."""
    results = {}
    for n in range(1, max_streams + 1):
        results[n] = float(bench_fn(n))

    knee_point = 1
    max_tp = 0.0
    for n in sorted(results.keys()):
        if results[n] > max_tp:
            max_tp = results[n]
            knee_point = n

    return results, knee_point


class OptimalStreamPool:
    """Stream pool that restricts concurrency to the optimal knee point."""

    def __init__(self, bench_fn, max_streams):
        self.bench_fn = bench_fn
        self.max_streams = max_streams
        self.results, self.knee_point = sweep_stream_count(bench_fn, max_streams)

    def get_optimal_streams(self):
        return self.knee_point

    def compute_throughput_ratio(self, oversubscribed_streams):
        opt_tp = self.results[self.knee_point]
        oversub_tp = self.results.get(oversubscribed_streams, self.bench_fn(oversubscribed_streams))
        if oversub_tp <= 0:
            return 1.0
        return opt_tp / oversub_tp
