def _simulate(workload, budget):
    """Reference continuous-batching scheduler simulation.

    Every iteration:
      - every currently-decoding request unconditionally gets 1 token
        (num_decoders of them),
      - whatever budget is left (``max(budget - num_decoders, 0)``) is
        spent continuing the single current prefill target (strict FCFS
        by (arrival, id); only one request prefills at a time),
      - the iteration's wall-clock duration equals the number of tokens it
        actually processed (num_decoders + prefill tokens consumed), and
        every token produced this iteration is timestamped at the new
        cumulative wall-clock time.

    Returns (max_ttft, max_itl) over the whole workload, where per-request
    TTFT = timestamp of its first produced token minus its arrival time,
    and per-request ITL = the largest gap between consecutive produced-token
    timestamps (0 if it only ever produces 0 or 1 tokens).
    """
    reqs = sorted(workload, key=lambda r: (r["arrival"], r["id"]))
    n = len(reqs)
    remaining_prompt = {r["id"]: r["prompt_len"] for r in reqs}
    remaining_decode = {r["id"]: r["decode_len"] for r in reqs}
    arrival = {r["id"]: r["arrival"] for r in reqs}
    history = {r["id"]: [] for r in reqs}
    active_decoders = []
    order = [r["id"] for r in reqs]  # FCFS prefill order

    total_decode_tokens = sum(r["decode_len"] for r in reqs)
    served_tokens = 0
    T = 0
    ptr = 0

    while served_tokens < total_decode_tokens:
        p = ptr
        target = None
        while p < n:
            rid = order[p]
            if remaining_prompt[rid] == 0:
                p += 1
                continue
            if arrival[rid] <= T:
                target = rid
            break
        ptr = p

        num_decoders = len(active_decoders)

        if target is None and num_decoders == 0:
            future = [arrival[order[i]] for i in range(ptr, n) if remaining_prompt[order[i]] > 0]
            if not future:
                break
            T = min(future)
            continue

        leftover = max(budget - num_decoders, 0)
        chunk = 0
        if target is not None and leftover > 0:
            chunk = min(leftover, remaining_prompt[target])
            remaining_prompt[target] -= chunk

        duration = num_decoders + chunk
        if duration == 0:
            future = [arrival[order[i]] for i in range(ptr, n) if remaining_prompt[order[i]] > 0]
            T = min(future)
            continue

        T = T + duration

        for rid in list(active_decoders):
            history[rid].append(T)
            remaining_decode[rid] -= 1
            served_tokens += 1
            if remaining_decode[rid] == 0:
                active_decoders.remove(rid)

        if chunk > 0 and remaining_prompt[target] == 0:
            active_decoders.append(target)

    max_ttft = 0
    max_itl = 0
    for r in reqs:
        rid = r["id"]
        hist = history[rid]
        ttft = hist[0] - arrival[rid]
        gaps = [hist[i + 1] - hist[i] for i in range(len(hist) - 1)]
        itl = max(gaps) if gaps else 0
        max_ttft = max(max_ttft, ttft)
        max_itl = max(max_itl, itl)

    return max_ttft, max_itl


def _oracle_choose(workload, candidate_budgets, ttft_cap):
    best_budget = None
    best_itl = None
    for b in candidate_budgets:
        max_ttft, max_itl = _simulate(workload, b)
        if max_ttft > ttft_cap:
            continue
        if best_itl is None or max_itl < best_itl or (max_itl == best_itl and b < best_budget):
            best_itl = max_itl
            best_budget = b
    return -1 if best_budget is None else best_budget


def _cases():
    workload1 = [
        {"id": 0, "arrival": 0, "prompt_len": 30, "decode_len": 6},
        {"id": 1, "arrival": 1, "prompt_len": 5, "decode_len": 8},
        {"id": 2, "arrival": 3, "prompt_len": 40, "decode_len": 6},
        {"id": 3, "arrival": 5, "prompt_len": 5, "decode_len": 8},
        {"id": 4, "arrival": 20, "prompt_len": 20, "decode_len": 5},
    ]
    workload2 = [
        {"id": 0, "arrival": 0, "prompt_len": 12, "decode_len": 5},
        {"id": 1, "arrival": 2, "prompt_len": 25, "decode_len": 6},
        {"id": 2, "arrival": 4, "prompt_len": 8, "decode_len": 7},
        {"id": 3, "arrival": 10, "prompt_len": 15, "decode_len": 4},
    ]
    return [
        (workload1, [4, 8, 16, 32, 64], 108),   # excludes budget=4 (ttft 109>108)
        (workload2, [2, 4, 8, 12, 16, 24, 32], 64),  # excludes 2,4,8
        (workload1, [4, 8, 16, 32, 64], 50),    # nothing feasible -> -1
    ]


def grade(sol, fx) -> dict:
    ok = 1.0
    for workload, candidates, cap in _cases():
        ref = _oracle_choose(workload, list(candidates), cap)
        try:
            got = sol.choose_budget_min_tail_itl(
                [dict(r) for r in workload], list(candidates), cap
            )
        except Exception:
            return {"exact_match": 0.0}
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
