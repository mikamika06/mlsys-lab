def extract_minimal_reproducer(raw_report: dict) -> dict:
    config = raw_report.get("config", {})
    minimal_config = {
        "model": config.get("model", "unknown"),
        "max_model_len": config.get("max_model_len", 2048),
        "tensor_parallel_size": config.get("tensor_parallel_size", 1)
    }
    return {
        "minimal_config": minimal_config,
        "trigger_prompt": raw_report.get("prompt", ""),
        "sampling_params": raw_report.get("sampling_params", {})
    }
