def token_bucket_policy(arrivals, capacity, refill_rate):
    tokens = float(capacity)
    last_time = 0.0
    accepted = []
    for t, req_id in arrivals:
        dt = t - last_time
        if dt > 0:
            tokens = min(float(capacity), tokens + dt * refill_rate)
            last_time = t
        if tokens >= 1.0:
            tokens -= 1.0
            accepted.append(req_id)
    return accepted


def concurrency_limit_policy(arrivals, max_concurrency):
    active = 0
    accepted = []
    for t, req_id, duration in arrivals:
        if active < max_concurrency:
            active += 1
            accepted.append((req_id, t, t + duration))
    return accepted


def delay_threshold_policy(arrivals, max_delay):
    queue = []
    accepted = []
    for t, req_id in arrivals:
        queue = [qt for qt in queue if qt > t]
        if len(queue) * 0.1 <= max_delay:
            queue.append(t + 1.0)
            accepted.append(req_id)
    return accepted
