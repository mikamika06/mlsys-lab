def compute_slot_context(
    ctx_size: int, n_parallel: int, model_max_ctx: int = 4096
) -> int:
    """Compute per-slot context length from total context and parallel slots."""
    raise NotImplementedError


def plan_slot_allocation(
    model_max_ctx: int, requested_c: int, requested_np: int, min_tokens_per_slot: int
) -> dict:
    """Plan slot allocation and validate against minimum per-slot token requirements."""
    raise NotImplementedError
