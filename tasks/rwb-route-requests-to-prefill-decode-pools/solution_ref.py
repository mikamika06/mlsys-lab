def route_prefill_decode(
    arrivals: list,
    prompt_lens: list,
    gen_lens: list,
    n_prefill_workers: int,
    n_decode_workers: int,
    t_prefill_per_token: float,
    t_decode_per_token: float,
):
    """
    A disaggregated-serving scheduler with separate prefill and decode
    worker pools. Requests are handled ONE AT A TIME in the given order:

    - Route the PREFILL phase to whichever prefill worker will be free
      soonest (least-loaded, i.e. smallest current "available_at" time;
      ties broken by the lowest worker index). It starts at
      max(worker's available_at, this request's arrival time) and takes
      prompt_lens[i] * t_prefill_per_token.
    - Once prefill finishes, route the DECODE phase the same way among
      decode workers, using the prefill-completion time as the earliest
      the decode phase can start. It takes gen_lens[i] * t_decode_per_token.

    Each worker's available_at is updated to the finish time of whatever
    it was just assigned, so later requests see the updated load.

    Returns (prefill_assignments, decode_assignments):
      prefill_assignments[p]: list of request indices routed to prefill
        worker p, in assignment order.
      decode_assignments[d]: list of request indices routed to decode
        worker d, in assignment order.
    """
    prefill_available = [0.0] * n_prefill_workers
    decode_available = [0.0] * n_decode_workers
    prefill_assignments = [[] for _ in range(n_prefill_workers)]
    decode_assignments = [[] for _ in range(n_decode_workers)]

    n = len(arrivals)
    for i in range(n):
        p = min(range(n_prefill_workers), key=lambda w: prefill_available[w])
        start = max(prefill_available[p], arrivals[i])
        finish = start + prompt_lens[i] * t_prefill_per_token
        prefill_available[p] = finish
        prefill_assignments[p].append(i)

        d = min(range(n_decode_workers), key=lambda w: decode_available[w])
        dstart = max(decode_available[d], finish)
        dfinish = dstart + gen_lens[i] * t_decode_per_token
        decode_available[d] = dfinish
        decode_assignments[d].append(i)

    return prefill_assignments, decode_assignments
