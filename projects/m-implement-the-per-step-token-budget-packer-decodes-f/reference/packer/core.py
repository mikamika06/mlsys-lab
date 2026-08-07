def pack_step(decodes, prefills, token_budget):
    scheduled_decodes = []
    scheduled_prefill_chunks = []
    remaining_budget = token_budget

    for d in decodes:
        if remaining_budget >= 1:
            scheduled_decodes.append(d)
            remaining_budget -= 1
        else:
            break

    active_prefill_idx = 0
    while remaining_budget > 0 and active_prefill_idx < len(prefills):
        p = prefills[active_prefill_idx]
        p_id = p["id"]
        p_len = p["remaining"]
        take = min(p_len, remaining_budget)
        scheduled_prefill_chunks.append({"id": p_id, "tokens": take})
        remaining_budget -= take
        if take == p_len:
            active_prefill_idx += 1
        else:
            prefills[active_prefill_idx]["remaining"] = p_len - take
            break

    return scheduled_decodes, scheduled_prefill_chunks


def compute_steps(prefill_length, token_budget, decode_count):
    if prefill_length <= 0:
        return 0
    effective_budget = token_budget - decode_count
    if effective_budget <= 0:
        return -1
    steps = 0
    rem = prefill_length
    while rem > 0:
        take = min(rem, effective_budget)
        rem -= take
        steps += 1
    return steps


def predict_itl_jitter(prefill_length, token_budget, decode_count, base_decode_latency_ms):
    if decode_count <= 0:
        return 0.0
    effective_budget = token_budget - decode_count
    if effective_budget <= 0:
        return float("inf")
    chunk_steps = compute_steps(prefill_length, token_budget, decode_count)
    if chunk_steps <= 0:
        return float("inf")
    unchunked_latency = (decode_count + prefill_length) * base_decode_latency_ms
    chunked_latency = base_decode_latency_ms * chunk_steps
    jitter_spike = unchunked_latency - chunked_latency
    return max(0.0, jitter_spike)
