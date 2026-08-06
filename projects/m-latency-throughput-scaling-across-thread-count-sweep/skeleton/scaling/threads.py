"""Thread scaling and oversubscription calculations."""


def find_oversubscription_point(topology, latency_data):
    """Derive oversubscription thread count given core topology and latency sweep data."""
    raise NotImplementedError


def analyze_thread_sweep(latency_data, work_units):
    """Compute throughput (ops/sec) and relative latency scaling for a thread count sweep."""
    raise NotImplementedError
