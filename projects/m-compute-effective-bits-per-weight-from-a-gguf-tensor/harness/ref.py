def get_tensor_fixtures():
    return [
        [
            {"n_elements": 1000000, "bits_per_weight": 4.5},
            {"n_elements": 2000000, "bits_per_weight": 4.0},
        ],
        [
            {"n_elements": 500000, "bits_per_weight": 8.0},
            {"n_elements": 500000, "bits_per_weight": 8.0},
        ],
        [
            {"n_elements": 123456, "bits_per_weight": 5.2},
            {"n_elements": 654321, "bits_per_weight": 4.8},
        ],
    ]


def get_perf_fixtures():
    return [
        [
            {"bpw": 4.5, "tok_s": 45.0},
            {"bpw": 6.0, "tok_s": 32.0},
            {"bpw": 8.0, "tok_s": 20.0},
        ]
    ]


def get_policy_fixtures():
    return (
        10.0,
        [
            {"name": "8B-Q8", "memory_gb": 9.5, "score": 88.0},
            {"name": "14B-Q4", "memory_gb": 8.2, "score": 92.0},
        ],
    )
