CONFIGS = [
    (
        {"weight_bytes": 1024 * 1024 * 1000, "kv_bytes": 1024 * 1024 * 200},
        {"weight_bytes": 1024 * 1024 * 800, "kv_bytes": 1024 * 1024 * 100},
        2000
    ),
    (
        {"weight_bytes": 1024 * 1024 * 3000, "kv_bytes": 1024 * 1024 * 500},
        {"weight_bytes": 1024 * 1024 * 2000, "kv_bytes": 1024 * 1024 * 500},
        4096
    ),
    (
        {"weight_bytes": 1024 * 1024 * 500, "kv_bytes": 1024 * 1024 * 100},
        {"weight_bytes": 1024 * 1024 * 400, "kv_bytes": 1024 * 1024 * 100},
        1024
    )
]

LIMIT_TESTS = [
    (
        {"weight_bytes": 1024 * 1024 * 1000, "kv_bytes": 1024 * 1024 * 200},
        {"weight_bytes": 1024 * 1024 * 1000, "kv_bytes": 1024 * 1024 * 200},
        [2048, 4096]
    )
]
