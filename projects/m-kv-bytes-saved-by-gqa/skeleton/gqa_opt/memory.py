def calculate_kv_cache_bytes(
    config: dict, batch_size: int, seq_len: int, dtype_bytes: int = 2
) -> dict:
    raise NotImplementedError


def analyze_gpu_expansion_overhead(
    config: dict, batch_size: int, seq_len: int, dtype_bytes: int = 2
) -> dict:
    raise NotImplementedError
