def compute_goodput(timestamps, latencies, admitted, slo_threshold, window):
    """Compute goodput: rate of admitted requests completing within the latency SLO.

    Returns requests-per-second of admitted requests whose measured latency
    does not exceed ``slo_threshold``, normalised by ``window``.
    """
    raise NotImplementedError('your code here')
