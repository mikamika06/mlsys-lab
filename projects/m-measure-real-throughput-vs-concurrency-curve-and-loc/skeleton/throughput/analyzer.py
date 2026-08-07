"""Concurrency curve analyzer and knee detector."""


def locate_knee(concurrency_levels, throughputs):
    """Locate the knee point on a throughput vs concurrency curve."""
    raise NotImplementedError


def evaluate_concurrency_capacity(concurrency_levels, throughputs, target_concurrency):
    """Evaluate throughput efficiency at target concurrency against maximum throughput."""
    raise NotImplementedError
