def compute_metrics(events):
    raise NotImplementedError

def consistency_error(throughput, mean_latency, concurrency):
    raise NotImplementedError

def validate_trace(events, concurrency, tol=0.05):
    raise NotImplementedError
