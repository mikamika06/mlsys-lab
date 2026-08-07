def pack_steps(prefill_lengths, budget, decodes_per_step):
    lengths = list(prefill_lengths)
    steps = []
    idx = 0
    while idx < len(lengths) or any(l > 0 for l in lengths):
        rem_budget = budget - decodes_per_step
        if rem_budget <= 0:
            break
        step_prefill = 0
        allocated = []
        while idx < len(lengths):
            needed = lengths[idx]
            if needed <= 0:
                idx += 1
                continue
            if step_prefill + needed <= rem_budget:
                step_prefill += needed
                allocated.append((idx, needed))
                lengths[idx] = 0
                idx += 1
            else:
                partial = rem_budget - step_prefill
                if partial > 0:
                    step_prefill += partial
                    lengths[idx] -= partial
                    allocated.append((idx, partial))
                break
        steps.append({"decodes": decodes_per_step, "prefill_allocated": allocated, "total_tokens": decodes_per_step + step_prefill})
        if idx >= len(lengths) and all(l <= 0 for l in lengths):
            break
    return steps


def compute_step_count(prefill_len, budget, decodes_per_step):
    rem_budget = budget - decodes_per_step
    if rem_budget <= 0:
        raise ValueError("Budget too small for decodes")
    steps = 0
    curr = prefill_len
    while curr > 0:
        curr -= rem_budget
        steps += 1
    return steps
