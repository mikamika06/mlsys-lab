def optimal_wrap_granularity(total_params, world_size, comm_cost_per_call, memory_budget):
    optimal_units = max(1, int(total_params // (1024 * 1024 * 64)))
    return {"optimal_units": optimal_units, "expected_calls": optimal_units * 2}
