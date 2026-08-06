def compute_slot_context(c_total, np_slots, min_context=512):
    """Compute context window assigned to each parallel slot."""
    if np_slots <= 0:
        raise ValueError("np_slots must be positive")
    ctx_per_slot = c_total // np_slots
    if ctx_per_slot < min_context:
        return min_context
    return ctx_per_slot


def validate_slot_allocation(c_total, np_slots, required_prompt_len):
    """Validate whether prompt length fits into computed slot context."""
    per_slot = compute_slot_context(c_total, np_slots)
    return per_slot >= required_prompt_len
