import math


def _batch_formation_times(arrivals: list[int], preferred_size: int, max_queue_delay: int) -> list[int]:
    """Simulate the same two-trigger dynamic batcher used elsewhere
    (size trigger == preferred_size, delay trigger == max_queue_delay,
    size wins on an exact tie), and return, for each request in
    ARRIVAL-SORTED order, the time its batch was dispatched."""
    order = sorted(range(len(arrivals)), key=lambda k: (arrivals[k], k))
    sorted_arrivals = [arrivals[k] for k in order]
    n = len(sorted_arrivals)

    formation = [0] * n
    i = 0
    queue: list[int] = []  # holds indices into sorted_arrivals
    while i < n or queue:
        next_arrival = sorted_arrivals[i] if i < n else None
        next_timeout = sorted_arrivals[queue[0]] + max_queue_delay if queue else None

        if next_arrival is not None and (next_timeout is None or next_arrival <= next_timeout):
            t = next_arrival
            while i < n and sorted_arrivals[i] == t:
                queue.append(i)
                i += 1
                if len(queue) == preferred_size:
                    for idx in queue:
                        formation[idx] = t
                    queue = []
        else:
            t = next_timeout
            for idx in queue:
                formation[idx] = t
            queue = []

    # undo the sort: formation[k] currently indexes sorted order; remap
    # back to the caller's original request order.
    out = [0] * n
    for pos, k in enumerate(order):
        out[k] = formation[pos]
    return out


def _p99(latencies: list[float]) -> float:
    """Nearest-rank p99: the smallest value such that at least 99% of the
    (sorted ascending) latencies are <= it, i.e. order statistic index
    ceil(0.99 * n) - 1 (0-indexed)."""
    n = len(latencies)
    s = sorted(latencies)
    k = math.ceil(0.99 * n)
    k = max(1, min(n, k))
    return s[k - 1]


def min_cap_meeting_slo(arrivals: list[int], candidate_caps: list[int],
                         preferred_size: int, service_time: int, slo: float) -> int:
    """Sweep candidate delay caps and return the SMALLEST one (by value,
    not list position) whose resulting p99 end-to-end latency meets the
    SLO, or -1 if none of them do.

    For a given cap C, every request's latency is
    (batch_formation_time - arrival_time) + service_time, using the
    two-trigger batcher (size == preferred_size, delay == C) over
    `arrivals`. `service_time` is a fixed per-batch compute cost applied
    identically to every request. p99 is the nearest-rank order
    statistic defined above.

    arrivals        : list of int arrival times (need not be pre-sorted).
    candidate_caps  : list of positive int candidate `max_queue_delay`
                       values (need not be pre-sorted; duplicates
                       possible).
    preferred_size  : positive int, the batcher's size trigger.
    service_time    : non-negative int, fixed compute time per batch.
    slo             : float, the p99 latency budget.

    Returns the smallest passing cap value, or -1 if no candidate meets
    the SLO.
    """
    best = None
    for cap in sorted(set(int(c) for c in candidate_caps)):
        formation = _batch_formation_times(arrivals, preferred_size, cap)
        latencies = [f - a + service_time for f, a in zip(formation, arrivals)]
        p99 = _p99(latencies)
        if p99 <= slo:
            best = cap
            break
    return best if best is not None else -1
