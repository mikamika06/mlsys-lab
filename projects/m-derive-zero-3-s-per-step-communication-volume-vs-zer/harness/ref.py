CONFIGS = [
    {"params_bytes": 1000000, "world_size": 4},
    {"params_bytes": 4000000, "world_size": 8},
    {"params_bytes": 16000000, "world_size": 16},
]

INIT_LOGS = [
    ["Rank 0: partition_size = 250000", "Rank 1: partition_size = 250000"],
    ["Rank 0: partition_size = 500000", "Rank 1: partition_size = 500000"],
]

RUNTIME_LOGS = [
    ["Rank 0: memory_reduction = 3.5", "Rank 1: memory_reduction = 3.5"],
    ["Rank 0: memory_reduction = 7.2", "Rank 1: memory_reduction = 7.2"],
]
