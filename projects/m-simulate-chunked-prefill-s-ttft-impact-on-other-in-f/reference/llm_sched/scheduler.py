def simulate_schedule(prompt_len, inflight_reqs, chunk_size, prefill_cost, decode_cost):
    time = 0.0
    max_stall = 0.0
    rem = prompt_len
    active = list(inflight_reqs)
    ttft = 0.0

    while rem > 0 or active:
        step_stall = 0.0
        if rem > 0:
            take = min(rem, chunk_size)
            step_stall = take * prefill_cost
            time += step_stall
            rem -= take
            if rem == 0:
                ttft = time

        max_stall = max(max_stall, step_stall)

        if active:
            time += len(active) * decode_cost
            active = [r - 1 for r in active if r > 1]

    return {"ttft": ttft, "max_stall": max_stall, "total_time": time}
