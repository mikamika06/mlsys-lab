def compute_overlap(request_tokens, worker_tokens):
    match_len = 0
    for r_t, w_t in zip(request_tokens, worker_tokens):
        if r_t == w_t:
            match_len += 1
        else:
            break
    return match_len


def compute_staleness(last_access_tick, current_tick, decay_factor):
    age = max(0, current_tick - last_access_tick)
    return float(age) * float(decay_factor)
