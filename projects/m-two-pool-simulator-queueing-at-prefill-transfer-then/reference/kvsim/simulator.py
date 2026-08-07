def simulate_pipeline(requests, prefill_capacity, decode_capacity, bandwidth_bps):
    prefill_queue = list(requests)
    prefill_active = []
    transfer_queue = []
    decode_queue = []
    decode_active = []
    completed = []
    t = 0.0

    while prefill_queue or prefill_active or transfer_queue or decode_queue or decode_active:
        while prefill_queue and len(prefill_active) < prefill_capacity:
            req = prefill_queue.pop(0)
            req["prefill_start"] = t
            req["prefill_end"] = t + req["prompt_tokens"] / 1000.0
            prefill_active.append(req)

        ready_prefill = [r for r in prefill_active if r["prefill_end"] <= t]
        for r in ready_prefill:
            transfer_queue.append(r)
        prefill_active = [r for r in prefill_active if r["prefill_end"] > t]

        if transfer_queue and not decode_active:
            req = transfer_queue.pop(0)
            req["transfer_start"] = t
            req["transfer_end"] = t + req["kv_size_bytes"] / float(bandwidth_bps)
            decode_queue.append(req)

        ready_transfer = [r for r in decode_queue if r.get("transfer_end", 0) <= t]
        for r in ready_transfer:
            if len(decode_active) < decode_capacity:
                r["decode_start"] = t
                r["decode_end"] = t + r["decode_tokens"] / 500.0
                decode_active.append(r)
                decode_queue.remove(r)

        decode_next = []
        for r in decode_active:
            if r["decode_end"] <= t:
                r["completion_time"] = t
                completed.append(r)
            else:
                decode_next.append(r)
        decode_active = decode_next

        t += 0.05
        if t > 500.0:
            break

    return completed
