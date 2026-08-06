RUN_A = {"threads": 4, "batch_size": 512, "ubatch_size": 256, "pp_throughput": 120.0}
RUN_B = {"threads": 8, "batch_size": 512, "ubatch_size": 256, "pp_throughput": 120.0}
DEFAULT_CONFIG = {"threads": 4, "batch_size": 512, "ubatch_size": 256, "pp_throughput": 100.0}
TG_RUNS = [
    {"bytes_read": 1000, "tg_throughput": 12.0},
    {"bytes_read": 2000, "tg_throughput": 18.0},
    {"bytes_read": 3000, "tg_throughput": 25.0},
]
