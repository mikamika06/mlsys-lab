def admission_order(gen_lens, slot_count):
    """Choose an admission order minimizing mean completion latency.

    gen_lens: list of N positive ints, generation length of each waiting
        request (index = request id), all present at t=0.
    slot_count: number of concurrent processing slots (S >= 1).

    Returns (order, mean_completion_latency):
      order: a permutation of range(N), the dispatch order.
      mean_completion_latency: the mean completion time actually produced
        by simulating `order` with list scheduling -- whenever a slot is
        free, dispatch the next request in `order` to whichever slot is
        free soonest; that request's completion time is
        (that slot's free time) + gen_lens[request].

    Your order must achieve the minimum possible mean completion latency.
    """
    raise NotImplementedError('your code here')
