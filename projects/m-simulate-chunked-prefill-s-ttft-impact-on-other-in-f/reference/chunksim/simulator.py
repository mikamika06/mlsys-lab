def simulate_chunked_prefill(requests, chunk_size, max_tokens_per_step):
    reqs = []
    for r in requests:
        reqs.append({
            "id": r["id"],
            "prompt_len": r["prompt_len"],
            "gen_len": r["gen_len"],
            "arrival": r["arrival"],
            "processed_prompt": 0,
            "generated": 0,
            "first_token_time": None,
            "finish_time": None,
            "type": "prefill"
        })

    time = 0
    active = []
    completed = []

    while reqs or active:
        while reqs and reqs[0]["arrival"] <= time:
            active.append(reqs.pop(0))

        if not active:
            if reqs:
                time = reqs[0]["arrival"]
                continue
            break

        tokens_budget = max_tokens_per_step
        step_work = []

        for r in active:
            if tokens_budget <= 0:
                break
            if r["type"] == "prefill":
                remaining = r["prompt_len"] - r["processed_prompt"]
                to_process = min(remaining, chunk_size, tokens_budget)
                if to_process > 0:
                    step_work.append((r, "prefill", to_process))
                    tokens_budget -= to_process
            else:
                if tokens_budget >= 1:
                    step_work.append((r, "decode", 1))
                    tokens_budget -= 1

        for r, wtype, count in step_work:
            if wtype == "prefill":
                r["processed_prompt"] += count
                if r["processed_prompt"] >= r["prompt_len"]:
                    r["first_token_time"] = time + 1
                    r["type"] = "decode"
            else:
                r["generated"] += count
                if r["generated"] >= r["gen_len"]:
                    r["finish_time"] = time + 1
                    active.remove(r)
                    completed.append(r)

        time += 1

    results = {}
    for r in completed:
        results[r["id"]] = {
            "ttft": r["first_token_time"] - r["arrival"],
            "latency": r["finish_time"] - r["arrival"]
        }
    return results
