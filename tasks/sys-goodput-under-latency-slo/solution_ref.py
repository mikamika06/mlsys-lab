def compute_goodput(timestamps, latencies, admitted, slo_threshold, window):
    """Compute goodput: rate of admitted requests completing within the latency SLO.

    Returns requests-per-second of admitted requests whose measured latency
    does not exceed ``slo_threshold``, normalised by ``window``.
    """
    count = 0.0
    for i in range(len(admitted)):
        if admitted[i] and latencies[i] <= slo_threshold:
            count += 1.0

    return float(count / window)
