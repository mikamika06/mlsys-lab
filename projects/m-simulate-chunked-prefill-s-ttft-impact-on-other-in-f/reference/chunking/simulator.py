def simulate_chunked_prefill(requests, chunk_size, token_time_ms):
    reqs = []
    for r in requests:
        reqs.append({
            "id": r["id"],
            "type": r["type"],
            "tokens_left": r["prompt_tokens"] if r["type"] == "prefill" else 1,
            "total_prompt": r["prompt_tokens"] if r["type"] == "prefill" else 0,
            "completion_tokens": r.get("completion_tokens", 16),
            "generated": 0,
            "start_time": r["arrival_time"],
            "ttft": None,
            "completion_time": None
        })

    time_cursor = 0.0
    active_requests = []
    completed = []

    all_incoming = sorted(reqs, key=lambda x: x["start_time"])
    incoming_idx = 0

    while incoming_idx < len(all_incoming) or active_requests:
        while incoming_idx < len(all_incoming) and all_incoming[incoming_idx]["start_time"] <= time_cursor:
            active_requests.append(all_incoming[incoming_idx])
            incoming_idx += 1

        if not active_requests:
            if incoming_idx < len(all_incoming):
                time_cursor = all_incoming[incoming_idx]["start_time"]
                continue
            break

        prefills = [r for r in active_requests if r["type"] == "prefill" and r["tokens_left"] > 0]
        decodes = [r for r in active_requests if r["type"] == "decode"]

        if prefills:
            p = prefills[0]
            chunk = min(chunk_size, p["tokens_left"])
            p["tokens_left"] -= chunk
            time_cursor += chunk * token_time_ms
            if p["tokens_left"] == 0:
                p["ttft"] = time_cursor - p["start_time"]
                p["type"] = "decode"
        elif decodes:
            time_cursor += token_time_ms
            for d in decodes:
                d["generated"] += 1
                if d["ttft"] is None:
                    d["ttft"] = time_cursor - d["start_time"]
                if d["generated"] >= d["completion_tokens"]:
                    d["completion_time"] = time_cursor
                    active_requests.remove(d)
                    completed.append(d)

        finished_prefills = [r for r in active_requests if r["type"] == "prefill" and r["tokens_left"] == 0]
        for fp in finished_prefills:
            pass

        removes = [r for r in active_requests if r["type"] == "decode" and r["completion_time"] is not None]
        for r in removes:
            if r in active_requests:
                active_requests.remove(r)

    results = []
    for r in sorted(completed + active_requests, key=lambda x: x["id"]):
        results.append({
            "id": r["id"],
            "ttft": r["ttft"] if r["ttft"] is not None else (time_cursor - r["start_time"]),
            "completion_time": r["completion_time"] if r["completion_time"] is not None else time_cursor
        })
    return results
