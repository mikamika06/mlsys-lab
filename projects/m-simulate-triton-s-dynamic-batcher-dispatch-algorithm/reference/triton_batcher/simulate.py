def simulate(arrivals: list[int], max_batch_size: int, preferred: list[int], max_delay_us: int, compute_us_fn) -> list[dict]:
    time_us = 0
    queue = []
    out = []
    next_req_idx = 0
    model_ready_us = 0

    preferred_sizes = sorted([p for p in preferred if p <= max_batch_size] + [max_batch_size], reverse=True)

    while next_req_idx < len(arrivals) or queue:
        while next_req_idx < len(arrivals) and arrivals[next_req_idx] <= time_us:
            queue.append(next_req_idx)
            next_req_idx += 1

        dispatched = False
        if model_ready_us <= time_us and queue:
            max_delay_reached = (time_us >= arrivals[queue[0]] + max_delay_us)
            chosen_batch_size = 0

            if max_delay_reached:
                chosen_batch_size = min(len(queue), max_batch_size)
            else:
                for p in preferred_sizes:
                    if len(queue) >= p:
                        chosen_batch_size = p
                        break

            if chosen_batch_size > 0:
                batch_reqs = queue[:chosen_batch_size]
                queue = queue[chosen_batch_size:]
                out.append({
                    "start_us": time_us,
                    "batch_size": chosen_batch_size,
                    "request_ids": batch_reqs
                })
                model_ready_us = time_us + compute_us_fn(chosen_batch_size)
                dispatched = True

        if not dispatched:
            candidates = []
            if next_req_idx < len(arrivals):
                candidates.append(arrivals[next_req_idx])
            if model_ready_us > time_us:
                candidates.append(model_ready_us)
            if model_ready_us <= time_us and queue:
                candidates.append(arrivals[queue[0]] + max_delay_us)

            if candidates:
                time_us = max(time_us, min(candidates))
            else:
                break

    return out
