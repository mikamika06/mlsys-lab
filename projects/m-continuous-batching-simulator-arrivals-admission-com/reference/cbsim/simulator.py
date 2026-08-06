class Request:
    def __init__(self, req_id, arrival_time, prompt_len, gen_len):
        self.req_id = req_id
        self.arrival_time = arrival_time
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.tokens_generated = 0
        self.start_time = None
        self.finish_time = None

def admission_predicate(active_requests, candidate_request, max_batch_size, max_capacity):
    if len(active_requests) >= max_batch_size:
        return False
    current_load = sum(r.prompt_len + r.tokens_generated for r in active_requests)
    candidate_load = candidate_request.prompt_len
    if current_load + candidate_load > max_capacity:
        return False
    return True

def simulate_continuous(requests, max_batch_size, max_capacity):
    sorted_reqs = sorted(requests, key=lambda r: r.arrival_time)
    pending = list(sorted_reqs)
    active = []
    completed = []
    time = 0
    while pending or active:
        while pending and pending[0].arrival_time <= time:
            req = pending[0]
            if admission_predicate(active, req, max_batch_size, max_capacity):
                pending.pop(0)
                req.start_time = max(time, req.arrival_time)
                active.append(req)
            else:
                break

        if not active and pending:
            time = max(time, pending[0].arrival_time)
            continue

        for req in list(active):
            req.tokens_generated += 1
            if req.tokens_generated >= req.gen_len:
                req.finish_time = time + 1
                active.remove(req)
                completed.append(req)
        time += 1
    return completed

def simulate_static(requests, max_batch_size, max_capacity):
    sorted_reqs = sorted(requests, key=lambda r: r.arrival_time)
    completed = []
    i = 0
    time = 0
    while i < len(sorted_reqs):
        batch = []
        max_len = 0
        while i < len(sorted_reqs) and len(batch) < max_batch_size:
            req = sorted_reqs[i]
            req_load = req.prompt_len + req.gen_len
            if req_load > max_capacity and not batch:
                batch.append(req)
                max_len = req.prompt_len + req.gen_len
                i += 1
                break
            elif not batch or sum(r.prompt_len for r in batch) + req.prompt_len <= max_capacity:
                batch.append(req)
                max_len = max(max_len, req.prompt_len + req.gen_len)
                i += 1
            else:
                break

        batch_start = max(time, max(r.arrival_time for r in batch))
        batch_end = batch_start + max_len
        for req in batch:
            req.start_time = batch_start
            req.finish_time = batch_end
            req.tokens_generated = req.gen_len
            completed.append(req)
        time = batch_end
    return completed

def compute_throughput_ratio(requests, max_batch_size, max_capacity):
    reqs_cont = [Request(r.req_id, r.arrival_time, r.prompt_len, r.gen_len) for r in requests]
    reqs_stat = [Request(r.req_id, r.arrival_time, r.prompt_len, r.gen_len) for r in requests]

    c_comp = simulate_continuous(reqs_cont, max_batch_size, max_capacity)
    s_comp = simulate_static(reqs_stat, max_batch_size, max_capacity)

    c_time = max(r.finish_time for r in c_comp) - min(r.arrival_time for r in c_comp)
    s_time = max(r.finish_time for r in s_comp) - min(r.arrival_time for r in s_comp)

    c_tokens = sum(r.prompt_len + r.gen_len for r in c_comp)
    s_tokens = sum(r.prompt_len + r.gen_len for r in s_comp)

    c_tp = c_tokens / c_time if c_time > 0 else 0
    s_tp = s_tokens / s_time if s_time > 0 else 0

    if s_tp == 0:
        return 1.0
    return c_tp / s_tp
