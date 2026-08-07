LOGS_TESTS = [
    (
        ["INFO: boot", "warning: flash-attention disabled", "warning: paged-kv disabled"],
        ["flash-attention", "paged-kv"]
    ),
    (
        ["silently disabled feature: fused-norm"],
        ["fused-norm"]
    )
]

BAKEOFF_TESTS = [
    (
        {"layers": 32, "kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "block_size": 16},
        {"layers": 32, "kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "block_size": 16},
        2000000
    )
]

MATRIX_TESTS = [
    (
        "class VLLMEngine:\n    def step(self):\n        pass\n",
        {"VLLMEngine": ["step"]}
    )
]
