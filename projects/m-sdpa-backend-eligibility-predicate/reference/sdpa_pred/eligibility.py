def is_eligible(backend: str, dtype: str, is_causal: bool, q_len: int, kv_len: int, head_dim: int, device_cap: tuple) -> bool:
    if backend == "flash_attention":
        if device_cap < (8, 0):
            return False
        if dtype not in ("float16", "bfloat16"):
            return False
        if head_dim > 128 or head_dim % 8 != 0:
            return False
        return True
    elif backend == "mem_efficient":
        if device_cap < (7, 0):
            return False
        if dtype not in ("float16", "bfloat16", "float32"):
            return False
        if head_dim > 256:
            return False
        return True
    elif backend == "math":
        return True
    return False
