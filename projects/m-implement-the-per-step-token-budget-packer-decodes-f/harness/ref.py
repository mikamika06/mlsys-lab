def pack_step(decodes_count, prefill_list, budget):
    used = decodes_count
    allocated_prefills = []
    remaining_prefills = list(prefill_list)
    rem_budget = budget - used
    while rem_budget > 0 and remaining_prefills:
        p = remaining_prefills.pop(0)
        if p <= rem_budget:
            allocated_prefills.append((0, p))
            rem_budget -= p
        else:
            allocated_prefills.append((0, rem_budget))
            remaining_prefills.insert(0, p - rem_budget)
            rem_budget = 0
    return decodes_count, allocated_prefills, remaining_prefills


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


def predict_itl_jitter(prefill_lens, budget, decodes_per_step, unchunked=False):
    if unchunked:
        latencies = [float(p + decodes_per_step) for p in prefill_lens]
        return max(latencies) if latencies else 0.0
    else:
        total_latencies = [float(compute_steps(p, budget, decodes_per_step)) for p in prefill_lens]
        return max(total_latencies) if total_latencies else 0.0


CONFIGS = [
    {"decodes": 4, "prefills": [128, 256, 512], "budget": 512},
    {"decodes": 10, "prefills": [1024, 64], "budget": 256},
]
