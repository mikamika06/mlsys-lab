def concurrency_ceiling(
    gpu_memory_gb: float,
    model_config: dict,
    tp_size: int,
    model_dtype: str,
    kv_dtype: str,
    seq_len: int,
    gpu_memory_utilization: float = 0.9,
) -> int:
    raise NotImplementedError

def build_feasibility_matrix(
    model_config: dict,
    target_seq_len: int,
    gpu_configs: list[dict],
    tp_options: list[int],
    model_dtypes: list[str],
    kv_dtypes: list[str],
    gpu_memory_utilization: float = 0.9,
) -> list[dict]:
    raise NotImplementedError
