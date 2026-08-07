def compute_steps(prefill_len, budget, decodes_per_step):
    steps = 0
    curr_len = prefill_len
    while curr_len > 0:
        avail = budget - decodes_per_step
        if avail <= 0:
            raise ValueError("Budget too small for decodes")
        chunk = min(curr_len, avail)
        curr_len -= chunk
        steps += 1
    return steps
