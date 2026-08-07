from kvcapacity.floor import model_weights_bytes, per_request_kv_bytes

def concurrency_ceiling(
    gpu_memory_gb: float,
    model_config: dict,
    tp_size: int,
    model_dtype: str,
    kv_dtype: str,
    seq_len: int,
    gpu_memory_utilization: float = 0.9,
) -> int:
    num_kv_heads = int(model_config.get("num_key_value_heads", model_config.get("num_attention_heads", 32)))
    if tp_size <= 0 or num_kv_heads % tp_size != 0:
        return 0
    total_gpu_bytes = gpu_memory_gb * (1024**3) * gpu_memory_utilization
    weight_bytes_per_gpu = model_weights_bytes(model_config, model_dtype) / tp_size
    available_kv_bytes = total_gpu_bytes - weight_bytes_per_gpu
    if available_kv_bytes <= 0:
        return 0
    full_req_kv_bytes = per_request_kv_bytes(model_config, seq_len, kv_dtype)
    kv_bytes_per_gpu = full_req_kv_bytes / tp_size
    if kv_bytes_per_gpu <= 0:
        return 0
    ceiling = int(available_kv_bytes // kv_bytes_per_gpu)
    return max(0, ceiling)

def build_feasibility_matrix(
    model_config: dict,
    target_seq_len: int,
    gpu_configs: list[dict],
    tp_options: list[int],
    model_dtypes: list[str],
    kv_dtypes: list[str],
    gpu_memory_utilization: float = 0.9,
) -> list[dict]:
    matrix = []
    for gpu in gpu_configs:
        for tp in tp_options:
            for m_dtype in model_dtypes:
                for k_dtype in kv_dtypes:
                    c = concurrency_ceiling(
                        gpu["memory_gb"],
                        model_config,
                        tp,
                        m_dtype,
                        k_dtype,
                        target_seq_len,
                        gpu_memory_utilization,
                    )
                    matrix.append({
                        "gpu_name": gpu["name"],
                        "gpu_memory_gb": gpu["memory_gb"],
                        "tp_size": tp,
                        "model_dtype": m_dtype,
                        "kv_dtype": k_dtype,
                        "seq_len": target_seq_len,
                        "concurrency_ceiling": c,
                        "feasible": c >= 1,
                    })
    return matrix
