def measure_reuse_overhead(cached, iterations):
    base_cost = 5.0 if cached else 50.0
    return base_cost * iterations
