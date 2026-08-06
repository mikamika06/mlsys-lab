def compute_preallocated_budget(
    num_layers: int,
    target_modules: list[str],
    hidden_shapes: dict[str, tuple[int, int]],
    max_loras: int,
    max_lora_rank: int,
    dtype_bytes: int = 2
) -> int:
    raise NotImplementedError


def can_fit_adapters(
    adapters: list[dict],
    base_model_shapes: dict,
    memory_cap_bytes: int
) -> bool:
    raise NotImplementedError
