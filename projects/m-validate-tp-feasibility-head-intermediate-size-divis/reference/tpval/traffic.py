def compute_tp_traffic(config: dict, tp_size: int, target_tokens_per_sec: float) -> dict:
    if tp_size <= 1:
        return {
            "bytes_per_token_per_rank": 0.0,
            "total_bus_bytes_per_sec": 0.0
        }

    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    dtype_bytes = config["dtype_bytes"]

    bytes_per_token_per_rank = 2 * (tp_size - 1) / tp_size * (2 * hidden_size * num_layers * dtype_bytes)
    total_bus_bytes_per_sec = bytes_per_token_per_rank * target_tokens_per_sec * tp_size

    return {
        "bytes_per_token_per_rank": float(bytes_per_token_per_rank),
        "total_bus_bytes_per_sec": float(total_bus_bytes_per_sec)
    }


def compute_pp_bubble_fraction(num_microbatches: int, num_pipeline_stages: int) -> float:
    if num_microbatches <= 0 or num_pipeline_stages <= 0:
        raise ValueError("Microbatches and pipeline stages must be positive integers.")
    if num_microbatches < num_pipeline_stages:
        raise ValueError("Microbatches must be greater than or equal to pipeline stages.")

    p = num_pipeline_stages
    m = num_microbatches
    return float((p - 1) / (m + p - 1))
