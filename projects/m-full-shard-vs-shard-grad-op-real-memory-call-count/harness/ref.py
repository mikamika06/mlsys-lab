CONFIGS = [
    {"strategy": "FULL_SHARD", "num_params": 10000000, "hidden_dim": 1024, "num_layers": 16, "world_size": 8},
    {"strategy": "SHARD_GRAD_OP", "num_params": 10000000, "hidden_dim": 1024, "num_layers": 16, "world_size": 8},
    {"strategy": "FULL_SHARD", "num_params": 50000000, "hidden_dim": 2048, "num_layers": 32, "world_size": 4},
    {"strategy": "SHARD_GRAD_OP", "num_params": 50000000, "hidden_dim": 2048, "num_layers": 32, "world_size": 4},
]

UNIT_TESTS = [
    {"num_layers": 24, "wrap_threshold_params": 5000000, "layer_param_count": 1000000},
    {"num_layers": 12, "wrap_threshold_params": 2000000, "layer_param_count": 500000},
]

OPTIMAL_TESTS = [
    {"total_params": 100000000, "world_size": 8, "comm_cost_per_call": 0.01, "memory_budget": 1024 * 1024 * 1024},
]
