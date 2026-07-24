def admission_order(gen_lens, slot_count):
    n = len(gen_lens)
    order = sorted(range(n), key=lambda i: gen_lens[i])

    slot_free = [0.0] * slot_count
    completions = [0.0] * n
    for i in order:
        j = min(range(slot_count), key=lambda x: slot_free[x])
        c = slot_free[j] + gen_lens[i]
        slot_free[j] = c
        completions[i] = c

    mean_completion_latency = sum(completions) / n
    return order, mean_completion_latency
