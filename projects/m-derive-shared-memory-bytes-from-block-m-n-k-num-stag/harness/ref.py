def get_m1_cases():
    return [
        {
            "block_m": 64,
            "block_n": 64,
            "block_k": 32,
            "num_stages": 3,
            "dtype": "float16",
            "expected_bytes": (64 * 32 + 32 * 64) * 2 * 3,
        },
        {
            "block_m": 128,
            "block_n": 64,
            "block_k": 32,
            "num_stages": 4,
            "dtype": "bfloat16",
            "expected_bytes": (128 * 32 + 32 * 64) * 2 * 4,
        },
        {
            "block_m": 64,
            "block_n": 128,
            "block_k": 32,
            "num_stages": 2,
            "dtype": "float32",
            "expected_bytes": (64 * 32 + 32 * 128) * 4 * 2,
        },
        {
            "block_m": 32,
            "block_n": 32,
            "block_k": 32,
            "num_stages": 3,
            "dtype": "int8",
            "expected_bytes": (32 * 32 + 32 * 32) * 1 * 3,
        },
    ]


def get_m2_cases():
    cands = [
        {"block_m": 64, "block_n": 64, "block_k": 32, "num_stages": 3, "dtype": "float16"},
        {"block_m": 128, "block_n": 128, "block_k": 32, "num_stages": 4, "dtype": "float16"},
        {"block_m": 32, "block_n": 32, "block_k": 32, "num_stages": 2, "dtype": "float16"},
    ]
    return [
        {
            "error_msg": "cudaErrorLaunchOutOfResources",
            "max_bytes": 49152,
            "candidates": cands,
            "expected": [cands[0], cands[2]],
        }
    ]
