KEY_ARGS = ["M", "N"]
CALL_SEQUENCES = [
    {"M": 1024, "N": 1024, "K": 512, "other": 1},
    {"M": 1024, "N": 1024, "K": 1024, "other": 2},
    {"M": 2048, "N": 1024, "K": 512, "other": 3},
    {"M": 2048, "N": 1024, "K": 2048, "other": 4},
]

SWEEP_RECORDS = [
    {"config": {"BLOCK_M": 16, "BLOCK_N": 16}, "latency": 45.2},
    {"config": {"BLOCK_M": 32, "BLOCK_N": 32}, "latency": 12.1},
    {"config": {"BLOCK_M": 64, "BLOCK_N": 64}, "latency": 18.5},
]

SEARCH_TIMES = [5.0, 5.0, 5.0]
HARDCODED_TIME = 2.5
