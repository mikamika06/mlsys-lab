def simulate_wasted_tokens(requests, preemptions):
    req_map = {r["id"]: r["prefill_len"] for r in requests}
    total_wasted = 0
    for p in preemptions:
        rid = p["request_id"]
        if rid in req_map:
            total_wasted += req_map[rid]
    return total_wasted
