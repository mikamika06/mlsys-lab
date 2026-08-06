def compute_kv_bytes(num_layers, num_kv_heads, head_dim, num_ctx, num_parallel, dtype_bytes=2):
    return 2 * num_layers * num_kv_heads * head_dim * num_ctx * num_parallel * dtype_bytes

def max_feasible_parallel(num_layers, num_kv_heads, head_dim, num_ctx, dtype_bytes, total_vram_bytes, weights_bytes):
    avail = total_vram_bytes - weights_bytes
    if avail <= 0:
        return 0
    bytes_per_slot = compute_kv_bytes(num_layers, num_kv_heads, head_dim, num_ctx, 1, dtype_bytes)
    return int(avail // bytes_per_slot)

def simulate(requests, num_parallel, max_queue_size, ms_per_token):
    running = []
    queue = []
    completed = []
    dropped = []
    latencies = {}
    total_tokens = 0

    req_idx = 0
    time = 0

    while req_idx < len(requests) or running or queue:
        next_arr = requests[req_idx]["arrival"] if req_idx < len(requests) else float('inf')
        next_fin = min([r[0] for r in running]) if running else float('inf')
        time = min(next_arr, next_fin)

        finished_this = [r for r in running if r[0] == time]
        running = [r for r in running if r[0] > time]

        for f in sorted(finished_this, key=lambda x: x[1]["id"]):
            req = f[1]
            completed.append(req["id"])
            latencies[req["id"]] = time - req["arrival"]
            total_tokens += req["tokens"]

        while queue and len(running) < num_parallel:
            q_req = queue.pop(0)
            running.append((time + q_req["tokens"] * ms_per_token, q_req))

        while req_idx < len(requests) and requests[req_idx]["arrival"] == time:
            req = requests[req_idx]
            req_idx += 1
            if len(running) < num_parallel:
                running.append((time + req["tokens"] * ms_per_token, req))
            elif len(queue) < max_queue_size:
                queue.append(req)
            else:
                dropped.append(req["id"])

    min_arrival = min([r["arrival"] for r in requests]) if requests else 0
    makespan = time - min_arrival
    agg_tok_s = (total_tokens / (makespan / 1000.0)) if makespan > 0 else 0.0

    return {
        "completed": completed,
        "dropped": dropped,
        "aggregate_tok_s": agg_tok_s,
        "latencies": latencies
    }
