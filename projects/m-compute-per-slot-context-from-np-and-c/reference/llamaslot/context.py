def compute_slot_context(
    ctx_size: int, n_parallel: int, model_max_ctx: int = 4096
) -> int:
    """Compute per-slot context length from total context and parallel slots."""
    if n_parallel <= 0:
        return 0
    effective_ctx = model_max_ctx if ctx_size == 0 else ctx_size
    return effective_ctx // n_parallel


def plan_slot_allocation(
    model_max_ctx: int, requested_c: int, requested_np: int, min_tokens_per_slot: int
) -> dict:
    """Plan slot allocation and validate against minimum per-slot token requirements."""
    effective_c = model_max_ctx if requested_c == 0 else requested_c
    if requested_np <= 0:
        return {
            "total_ctx": effective_c,
            "n_parallel": requested_np,
            "slot_ctx": 0,
            "is_valid": False,
        }
    slot_ctx = effective_c // requested_np
    is_valid = (
        (requested_np > 0)
        and (slot_ctx >= min_tokens_per_slot)
        and (effective_c <= model_max_ctx)
    )
    return {
        "total_ctx": effective_c,
        "n_parallel": requested_np,
        "slot_ctx": slot_ctx,
        "is_valid": is_valid,
    }
