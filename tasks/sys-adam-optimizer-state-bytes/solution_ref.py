def adam_optimizer_state_bytes(num_params: int, mixed_precision: bool) -> int:
    bytes_per_param = 8
    if mixed_precision:
        bytes_per_param += 4
    return num_params * bytes_per_param
