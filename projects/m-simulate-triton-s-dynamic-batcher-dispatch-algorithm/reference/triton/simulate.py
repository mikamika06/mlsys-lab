def simulate(arrivals: list[int], max_batch_size: int, preferred_batch_sizes: list[int], max_queue_delay_us: int, compute_fn) -> list[dict]:
    reqs = [(i, arr) for i, arr in enumerate(arrivals)]
    reqs.sort(key=lambda x: x[1])

    Q = []
    out = []
    t = 0
    model_ready = 0

    preferred = sorted(list(set(preferred_batch_sizes + [max_batch_size])), reverse=True)

    while reqs or Q:
        while reqs and reqs[0][1] <= t:
            Q.append(reqs.pop(0))

        can_dispatch = False
        if Q and model_ready <= t:
            if t >= Q[0][1] + max_queue_delay_us:
                can_dispatch = True
            elif any(len(Q) >= p for p in preferred):
                can_dispatch = True

        if can_dispatch:
            b = 0
            if t >= Q[0][1] + max_queue_delay_us:
                b = min(len(Q), max_batch_size)
            else:
                for p in preferred:
                    if len(Q) >= p:
                        b = p
                        break

            batch = Q[:b]
            Q = Q[b:]

            out.append({
                "start_time": t,
                "batch_size": b,
                "request_ids": [req[0] for req in batch]
            })
            model_ready = t + compute_fn(b)
        else:
            next_t = float('inf')
            if reqs:
                next_t = min(next_t, float(reqs[0][1]))
            if model_ready > t:
                next_t = min(next_t, float(model_ready))
            if Q:
                next_t = min(next_t, float(Q[0][1] + max_queue_delay_us))

            if next_t == float('inf'):
                break
            t = int(next_t)

    return out
