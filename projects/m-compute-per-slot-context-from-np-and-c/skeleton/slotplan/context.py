def compute_slot_context(c_total, np_slots, min_context=512):
    """Compute context window assigned to each parallel slot."""
    raise NotImplementedError


def validate_slot_allocation(c_total, np_slots, required_prompt_len):
    """Validate whether prompt length fits into computed slot context."""
    raise NotImplementedError
