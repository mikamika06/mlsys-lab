TRIAGE_CASES = [
    {
        "error_msg": "CUDA error: misaligned address",
        "cause": "Pointer arithmetic offset is not a multiple of vector alignment size.",
        "fix": "Ensure tensor strides and memory allocations are aligned to 16 or 32 bytes."
    },
    {
        "error_msg": "FlashAttention kernel launch failed: smem size exceeds hardware limit",
        "cause": "Requested shared memory per block exceeds the maximum capacity of the streaming multiprocessor.",
        "fix": "Reduce block size, reduce head dimension, or lower query/key sequence chunk sizes."
    },
    {
        "error_msg": "RuntimeError: oob access in flash_fwd_kernel",
        "cause": "Sequence length or batch dimensions do not match tensor stride expectations, causing out-of-bounds index calculation.",
        "fix": "Verify attention mask dimensions and ensure padding tokens are correctly masked out."
    },
    {
        "error_msg": "CUDA error: invalid configuration argument",
        "cause": "Grid or block dimensions passed to kernel execution exceed device limits.",
        "fix": "Check batch size and number of attention heads configuration parameters."
    },
    {
        "error_msg": "FlashAttention error: head_dim must be a multiple of 32",
        "cause": "Head dimension is not aligned with tensor core requirements.",
        "fix": "Pad head dimension to a multiple of 32."
    }
]

VALIDATOR_CONFIGS = [
    {"config": {"head_dim": 64, "block_m": 64, "block_n": 64}, "valid": True},
    {"config": {"head_dim": 48, "block_m": 64, "block_n": 64}, "valid": False},
    {"config": {"head_dim": 128, "block_m": 32, "block_n": 32}, "valid": True},
    {"config": {"head_dim": 16, "block_m": 64, "block_n": 64}, "valid": False},
    {"config": {"head_dim": 96, "block_m": 64, "block_n": 64}, "valid": True}
]
