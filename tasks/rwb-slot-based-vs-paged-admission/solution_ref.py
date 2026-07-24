def admission_capacity(memory_budget, n_ctx, block_size, request_lengths):
    slot_used = 0
    slot_capacity = 0
    for _ in request_lengths:
        slot_used += n_ctx
        if slot_used <= memory_budget:
            slot_capacity += 1
        else:
            break

    paged_used = 0
    paged_capacity = 0
    for length in request_lengths:
        blocks = (length + block_size - 1) // block_size
        paged_used += blocks * block_size
        if paged_used <= memory_budget:
            paged_capacity += 1
        else:
            break

    return slot_capacity, paged_capacity
