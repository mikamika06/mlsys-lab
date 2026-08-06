from loraserve.config import compute_adapter_bytes


def compute_preallocated_budget(
    num_layers: int,
    target_modules: list[str],
    hidden_shapes: dict[str, tuple[int, int]],
    max_loras: int,
    max_lora_rank: int,
    dtype_bytes: int = 2
) -> int:
    per_adapter_params = 0
    for mod in target_modules:
        if mod in hidden_shapes:
            in_dim, out_dim = hidden_shapes[mod]
            per_adapter_params += (in_dim * max_lora_rank) + (max_lora_rank * out_dim)

    total_params = per_adapter_params * num_layers * max_loras
    return total_params * dtype_bytes


def can_fit_adapters(
    adapters: list[dict],
    base_model_shapes: dict,
    memory_cap_bytes: int
) -> bool:
    total_bytes = sum(compute_adapter_bytes(cfg, base_model_shapes) for cfg in adapters)
    return total_bytes <= memory_cap_bytes
