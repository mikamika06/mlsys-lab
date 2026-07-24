def gpipe_schedule(p: int, m: int) -> dict:
    """Simulate GPipe's fill-drain pipeline schedule for p devices (stages)
    and m microbatches, assuming every forward and every backward task takes
    exactly 1 discrete integer time slot.

    Each device i executes its 2*m tasks strictly in the order
    F(0), F(1), ..., F(m-1), B(m-1), B(m-2), ..., B(0)
    (all forwards, then all backwards -- this "fill then drain" ordering is
    what makes GPipe's schedule have bubbles, unlike 1F1B schedules). A task
    starts at the earliest integer slot such that:

    - device-order dependency: the same device's previous task (in the
      ordered list above) has already finished;
    - pipeline dependency for F(j) on device i>0: F(j) on device i-1 has
      finished (needs the activation from the previous stage);
    - pipeline dependency for B(j) on device i<p-1: B(j) on device i+1 has
      finished (needs the gradient from the next stage);
    - device 0's forwards and device (p-1)'s backwards have no pipeline
      dependency.

    Returns
    -------
    dict:
      timeline : list of p lists of (start, end) int tuples, one list per
        device, tasks in that device's execution order
        [F(0), F(1), ..., F(m-1), B(m-1), ..., B(0)].
      makespan : int -- the time slot at which the last task anywhere ends.
      bubble_slots : int -- total idle device-slots,
        p * makespan - p * 2 * m (sum, over all devices, of idle slots).
    """
    start = {}
    end = {}

    for j in range(m):
        for i in range(p):
            dev_prev = end[(i, "F", j - 1)] if j > 0 else 0
            pipe_prev = end[(i - 1, "F", j)] if i > 0 else 0
            s = max(dev_prev, pipe_prev)
            start[(i, "F", j)] = s
            end[(i, "F", j)] = s + 1

    for j in reversed(range(m)):
        for i in reversed(range(p)):
            if j == m - 1:
                dev_prev = end[(i, "F", m - 1)]
            else:
                dev_prev = end[(i, "B", j + 1)]
            pipe_prev = end[(i + 1, "B", j)] if i < p - 1 else 0
            s = max(dev_prev, pipe_prev)
            start[(i, "B", j)] = s
            end[(i, "B", j)] = s + 1

    makespan = max(end.values())
    bubble_slots = p * makespan - p * 2 * m

    timeline = []
    for i in range(p):
        seq = [(start[(i, "F", j)], end[(i, "F", j)]) for j in range(m)]
        seq += [(start[(i, "B", j)], end[(i, "B", j)]) for j in reversed(range(m))]
        timeline.append(seq)

    return {"timeline": timeline, "makespan": makespan, "bubble_slots": bubble_slots}
