def get_parameter_residency(model_size_bytes: int, strategy: str, phase: str) -> int:
    if phase == "between_forward":
        if strategy == "FULL_SHARD":
            return int(model_size_bytes // 4)
        elif strategy == "SHARD_GRAD_OP":
            return int(model_size_bytes)
    return int(model_size_bytes)
