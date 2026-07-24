import numpy as np

from mlsys import scorers


def _dispatch_times(arrivals, preferred_batch_size, max_queue_delay):
    """Real oracle: event-driven dynamic-batcher simulation. Two triggers,
    whichever fires first dispatches the whole current queue:
      - size trigger: queue reaches preferred_batch_size,
      - delay trigger: the oldest queued item has waited max_queue_delay.
    On an exact tie the size trigger wins (a full batch, not a partial one).

    Returns a list, same length and order as `arrivals`, of each request's
    dispatch timestamp.
    """
    arrivals = sorted(int(a) for a in arrivals)
    n = len(arrivals)
    i = 0
    queue = []  # list of (arrival_time, original_index)
    dispatch = [None] * n

    while i < n or queue:
        next_arrival = arrivals[i] if i < n else None
        next_timeout = queue[0][0] + max_queue_delay if queue else None

        if next_arrival is not None and (next_timeout is None or next_arrival <= next_timeout):
            t = next_arrival
            while i < n and arrivals[i] == t:
                queue.append((arrivals[i], i))
                i += 1
                if len(queue) == preferred_batch_size:
                    for _, idx in queue:
                        dispatch[idx] = t
                    queue = []
        else:
            t = next_timeout
            for _, idx in queue:
                dispatch[idx] = t
            queue = []

    return arrivals, dispatch


def _oracle_mean_latency(arrivals, preferred_batch_size, max_queue_delay) -> float:
    sorted_arrivals, dispatch = _dispatch_times(arrivals, preferred_batch_size, max_queue_delay)
    latencies = [d - a for a, d in zip(sorted_arrivals, dispatch)]
    return float(sum(latencies) / len(latencies))


def _synthetic_cases():
    rng = np.random.default_rng(101)
    cases = []
    for _ in range(5):
        n = int(rng.integers(6, 30))
        gaps = rng.integers(0, 8, size=n)
        arrivals = np.cumsum(gaps).tolist()
        preferred_batch_size = int(rng.integers(2, 8))
        max_queue_delay = int(rng.integers(3, 15))
        cases.append((arrivals, preferred_batch_size, max_queue_delay))
    return cases


def grade(sol, fx) -> dict:
    fixture_case = (fx["arrivals"].tolist(), 5, 6)
    cases = [fixture_case] + _synthetic_cases()

    worst = 0.0
    for arrivals, preferred_batch_size, max_queue_delay in cases:
        ref = _oracle_mean_latency(arrivals, preferred_batch_size, max_queue_delay)
        try:
            got = sol.mean_added_queue_latency(list(arrivals), preferred_batch_size, max_queue_delay)
            got = float(got)
        except Exception:
            return {"rel_err": float("inf")}

        if not np.isfinite(got):
            return {"rel_err": float("inf")}

        err = scorers.rel_err(np.array([ref]), np.array([got]))
        worst = max(worst, err)

    return {"rel_err": worst}
