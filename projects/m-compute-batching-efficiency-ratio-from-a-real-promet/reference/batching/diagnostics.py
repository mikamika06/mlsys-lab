from batching.latency import decompose_latency


def diagnose_throughput_drop(baseline_dump, current_dump):
    """Diagnose whether throughput drop is due to queueing or compute."""
    b_dec = decompose_latency(baseline_dump)
    c_dec = decompose_latency(current_dump)
    if c_dec["queue_fraction"] > b_dec["queue_fraction"]:
        return "queueing"
    return "compute"
