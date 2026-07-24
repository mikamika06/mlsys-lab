import numpy as np

def compute_goodput(timestamps, latencies, admitted, slo_threshold, window):
    """Compute goodput: rate of admitted requests completing within the latency SLO.

    Returns requests-per-second of admitted requests whose measured latency
    does not exceed ``slo_threshold``, normalised by ``window``.
    """
    # BUG: this computes throughput of SLO-meeting requests, but ignores the
    # admission mask.  A request that was rejected but happened to have a low
    # latency should NOT count toward goodput.
    latencies = np.asarray(latencies, dtype=np.float64)
    count = np.sum(latencies <= slo_threshold)
    return float(count / window)
