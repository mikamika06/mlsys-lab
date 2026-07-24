def _simulate(workload, budget):
    """Continuous-batching scheduler simulation for one candidate budget.

    Every iteration: every currently-decoding request unconditionally
    receives 1 token; whatever budget remains after that
    (max(budget - num_decoders, 0)) is spent continuing the single
    current prefill target (strict FCFS by (arrival, id), one request
    prefilling at a time). The iteration's wall-clock duration equals the
    number of tokens it actually processed, and every token produced this
    iteration is timestamped at the new cumulative wall-clock time.

    Returns (max_ttft, max_itl) over the whole workload.
    """
    reqs = sorted(workload, key=lambda r: (r["arrival"], r["id"]))
    n = len(reqs)
    remaining_prompt = {r["id"]: r["prompt_len"] for r in reqs}
    remaining_decode = {r["id"]: r["decode_len"] for r in reqs}
    arrival = {r["id"]: r["arrival"] for r in reqs}
    history = {r["id"]: [] for r in reqs}
    active_decoders = []
    order = [r["id"] for r in reqs]

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


def choose_budget_min_tail_itl(workload, candidate_budgets, ttft_cap):
    """
    Sweep every candidate budget, simulate the scheduler, and return the
    candidate that minimizes the worst-case (tail) inter-token latency
    among those whose worst-case TTFT satisfies the cap. Ties broken by
    the smallest budget. -1 if no candidate satisfies the cap.
    """
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
