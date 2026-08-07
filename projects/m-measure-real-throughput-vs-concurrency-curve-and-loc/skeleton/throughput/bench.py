"""Concurrency curve benchmarking."""


def simulate_serving_step(concurrency, num_requests, workload_spec):
    """Simulate serving execution for given concurrency and workload spec."""
    raise NotImplementedError


def measure_concurrency_curve(concurrency_levels, num_requests, workload_spec):
    """Measure throughput across a range of concurrency levels."""
    raise NotImplementedError
