class Request:
    def __init__(self, req_id, arrival_time, prompt_len, gen_len):
        self.req_id = req_id
        self.arrival_time = arrival_time
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.tokens_generated = 0
        self.start_time = None
        self.finish_time = None


def simulate_continuous_batching(trace, max_batch_size):
    reqs = [Request(r["id"], r["arrival"], r["prompt"], r["gen"]) for r in trace]
    reqs.sort(key=lambda x: x.arrival_time)
    
    time = 0
    active = []
    queue = []
    completed = []
    log = []
    
    req_idx = 0
    while req_idx < len(reqs) or active or queue:
        while req_idx < len(reqs) and reqs[req_idx].arrival_time <= time:
            queue.append(reqs[req_idx])
            req_idx += 1
            
        while queue and len(active) < max_batch_size:
            r = queue.pop(0)
            r.start_time = time
            active.append(r)
            
        log.append({"time": time, "active_count": len(active)})
        
        still_active = []
        for r in active:
            r.tokens_generated += 1
            if r.tokens_generated >= r.gen_len:
                r.finish_time = time + 1
                completed.append(r)
            else:
                still_active.append(r)
        active = still_active
        time += 1
        
    return completed, log


def compute_static_throughput_ratio(trace, max_batch_size):
    reqs = [Request(r["id"], r["arrival"], r["prompt"], r["gen"]) for r in trace]
    reqs.sort(key=lambda x: x.arrival_time)
    
    time = 0
    i = 0
    cont_complete_time = 0
    while i < len(reqs) or time <= (cont_complete_time if 'cont_complete_time' in locals() else 0):
        if i >= len(reqs):
            break
        batch = reqs[i:i+max_batch_size]
        max_len = max(r.prompt_len + r.gen_len for r in batch)
        time = max(time, batch[0].arrival_time) + max_len
        i += max_batch_size
    static_total_time = time
    
    completed, _ = simulate_continuous_batching(trace, max_batch_size)
    cont_total_time = max(r.finish_time for r in completed) - min(r.arrival_time for r in completed)
    
    if cont_total_time == 0:
        return 1.0
    return float(static_total_time) / float(cont_total_time)
