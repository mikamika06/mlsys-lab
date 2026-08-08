def get_test_cases():
    return {
        "numel": 1048576,
        "bytes_per_elem": 4,
        "world_size": 4,
        "log_lines": [
            "MEM_REPORT rank=0 z1=850000000 z3=320000000",
            "MEM_REPORT rank=1 z1=850000000 z3=320000000"
        ],
        "init_lines": [
            "PARAM_INIT name=model.layers.0.weight numel=4096",
            "PARAM_INIT name=model.layers.1.weight numel=2049"
        ]
    }
