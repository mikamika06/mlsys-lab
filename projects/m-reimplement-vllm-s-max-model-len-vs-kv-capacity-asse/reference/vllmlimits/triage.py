def triage_startup_error(error_msg, current_config):
    if "max_model_len" in error_msg or "KV cache capacity" in error_msg:
        return {"action": "reduce_max_model_len", "recommended_max_model_len": current_config["max_model_len"] // 2}
    elif "Out of memory" in error_msg:
        return {"action": "increase_gpu_memory_utilization", "recommended_gpu_memory_utilization": 0.95}
    return {"action": "none", "recommended_max_model_len": current_config["max_model_len"]}
