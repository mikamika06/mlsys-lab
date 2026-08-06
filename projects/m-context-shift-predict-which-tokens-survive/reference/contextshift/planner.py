from contextshift.match import find_surviving_tokens

def cache_survival_plan(old_tokens, new_tokens, block_size):
    """Generate cache block survival plan."""
    surviving_indices = set(find_surviving_tokens(old_tokens, new_tokens))
    total_blocks = (len(new_tokens) + block_size - 1) // block_size
    plan = []
    for b in range(total_blocks):
        start = b * block_size
        end = min(start + block_size, len(new_tokens))
        block_tokens = range(start, end)
        survives = all(idx in surviving_indices for idx in block_tokens)
        plan.append({"block_id": b, "survives": survives})
    return plan
