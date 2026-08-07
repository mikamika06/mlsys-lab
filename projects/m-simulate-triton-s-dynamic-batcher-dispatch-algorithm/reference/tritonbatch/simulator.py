def simulate_batcher(requests, max_batch_size, max_queue_delay_microseconds, preferred_batch_sizes=None):
    if preferred_batch_sizes is None:
        preferred_batch_sizes = []
    sorted_reqs = sorted(requests, key=lambda r: r["arrival_time"])
    if not sorted_reqs:
        return []

    queue = []
    batches = []
    req_idx = 0
    current_time = sorted_reqs[0]["arrival_time"]

    while req_idx < len(sorted_reqs) or queue:
        while req_idx < len(sorted_reqs) and sorted_reqs[req_idx]["arrival_time"] <= current_time:
            queue.append(sorted_reqs[req_idx])
            req_idx += 1

        if not queue:
            current_time = sorted_reqs[req_idx]["arrival_time"]
            continue

        oldest_arrival = queue[0]["arrival_time"]
        elapsed = current_time - oldest_arrival

        dispatch = False
        if len(queue) >= max_batch_size:
            dispatch = True
        elif elapsed >= max_queue_delay_microseconds:
            dispatch = True
        elif preferred_batch_sizes and len(queue) in preferred_batch_sizes:
            dispatch = True
        elif req_idx >= len(sorted_reqs) and queue:
            dispatch = True

        if dispatch:
            batch_size = min(max_batch_size, len(queue))
            batch_reqs = queue[:batch_size]
            queue = queue[batch_size:]
            batch_info = {
                "dispatch_time": current_time,
                "requests": batch_reqs,
                "size": len(batch_reqs),
                "max_wait": current_time - batch_reqs[0]["arrival_time"]
            }
            batches.append(batch_info)
        else:
            if req_idx < len(sorted_reqs):
                next_arrival = sorted_reqs[req_idx]["arrival_time"]
                timeout_time = oldest_arrival + max_queue_delay_microseconds
                current_time = min(next_arrival, timeout_time)
                if current_time <= oldest_arrival:
                    current_time = oldest_arrival + 1
            else:
                current_time = oldest_arrival + max_queue_delay_microseconds
    return batches
