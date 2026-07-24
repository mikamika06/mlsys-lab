import numpy as np


def _oracle(arrivals, preferred_batch_size, max_queue_delay):
    arrivals = sorted(int(a) for a in arrivals)
    n = len(arrivals)
    i = 0
    queue = []
    times = []
    sizes = []

    while i < n or queue:
        next_arrival = arrivals[i] if i < n else None
        next_timeout = queue[0] + max_queue_delay if queue else None

        if next_arrival is not None and (next_timeout is None or next_arrival <= next_timeout):
            t = next_arrival
            while i < n and arrivals[i] == t:
                queue.append(arrivals[i])
                i += 1
                if len(queue) == preferred_batch_size:
                    times.append(t)
                    sizes.append(preferred_batch_size)
                    queue = []
        else:
            t = next_timeout
            times.append(t)
            sizes.append(len(queue))
            queue = []

    return times, sizes


def _synthetic_cases():
    rng = np.random.default_rng(99)
    cases = []
    for _ in range(4):
        n = int(rng.integers(6, 25))
        gaps = rng.integers(0, 7, size=n)
        arrivals = np.cumsum(gaps).tolist()
        preferred_batch_size = int(rng.integers(2, 6))
        max_queue_delay = int(rng.integers(3, 12))
        cases.append((arrivals, preferred_batch_size, max_queue_delay))
    return cases


def grade(sol, fx) -> dict:
    fixture_case = (fx["arrivals"].tolist(), 5, 6)
    cases = [fixture_case] + _synthetic_cases()

    for arrivals, preferred_batch_size, max_queue_delay in cases:
        ref_times, ref_sizes = _oracle(arrivals, preferred_batch_size, max_queue_delay)

        try:
            got_times, got_sizes = sol.dynamic_batcher_simulate(
                list(arrivals), preferred_batch_size, max_queue_delay
            )
            got_times = [int(x) for x in got_times]
            got_sizes = [int(x) for x in got_sizes]
        except Exception:
            return {"exact_match": 0.0}

        if got_times != ref_times or got_sizes != ref_sizes:
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
