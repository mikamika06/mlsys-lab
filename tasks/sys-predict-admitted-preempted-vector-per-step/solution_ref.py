def schedule_admit_preempt(arrivals, lengths, priorities, budget, num_steps):
    """Simulate a preemptive-priority continuous-batching scheduler with a
    fixed running-set capacity `budget` over `num_steps` steps.

    `arrivals`, `lengths`, `priorities` are same-length lists of ints, one
    entry per request (request id == its index). Lower `priorities` value
    means higher priority.

    At every step the running set is recomputed from scratch as the
    `budget` best-priority requests among everyone who has arrived
    (`arrivals[i] <= t`) and has not yet finished (remaining length > 0),
    ties broken by request id. Running requests make one step of progress
    (their remaining length decreases by 1); once it hits 0 the request is
    finished and drops out of consideration.

    Returns a list of length `num_steps`. Step `t`'s entry is a dict:
      {"admitted": [...], "preempted": [...]}
    - "admitted": sorted request ids newly in the running set at step `t`
      that were not running at step `t-1`.
    - "preempted": sorted request ids that were running at step `t-1`,
      still have work left, but are not in the running set at step `t`.
    """
    n = len(arrivals)
    remaining = list(lengths)
    prev_running = set()
    steps = []

    for t in range(num_steps):
        active_pool = [i for i in range(n) if arrivals[i] <= t and remaining[i] > 0]
        ordered = sorted(active_pool, key=lambda i: (priorities[i], i))
        new_running = set(ordered[:budget])

        admitted = sorted(new_running - prev_running)
        preempted = sorted((prev_running & set(active_pool)) - new_running)
        steps.append({"admitted": admitted, "preempted": preempted})

        for i in new_running:
            remaining[i] -= 1
        prev_running = new_running

    return steps
