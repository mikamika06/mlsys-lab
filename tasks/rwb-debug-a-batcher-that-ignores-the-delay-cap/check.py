import numpy as np


def _oracle(arrivals, preferred_size, max_queue_delay):
    arrivals = sorted(int(a) for a in arrivals)
    n = len(arrivals)
    i = 0
    queue = []
    batches = []

    while i < n or queue:
        next_arrival = arrivals[i] if i < n else None
        next_timeout = queue[0] + max_queue_delay if queue else None

        if next_arrival is not None and (next_timeout is None or next_arrival <= next_timeout):
            t = next_arrival
            while i < n and arrivals[i] == t:
                queue.append(arrivals[i])
                i += 1
                if len(queue) == preferred_size:
                    batches.append((t, preferred_size))
                    queue = []
        else:
            t = next_timeout
            batches.append((t, len(queue)))
            queue = []

    return batches


def _synthetic_cases():
    rng = np.random.default_rng(43)
    cases = []
    for _ in range(4):
        n = int(rng.integers(6, 25))
        gaps = rng.integers(0, 6, size=n)
        arrivals = np.cumsum(gaps).tolist()
        preferred_size = int(rng.integers(2, 6))
        max_queue_delay = int(rng.integers(3, 12))
        cases.append((arrivals, preferred_size, max_queue_delay))
    return cases


def grade(sol, fx) -> dict:
    fixture_case = (fx["arrivals"].tolist(), 4, 10)
    cases = [fixture_case] + _synthetic_cases()

    total = 0
    correct = 0
    for arrivals, preferred_size, max_queue_delay in cases:
        ref = _oracle(arrivals, preferred_size, max_queue_delay)
        total += len(ref)
        try:
            got = sol.batch_formation_times(list(arrivals), preferred_size, max_queue_delay)
        except Exception:
            continue

        try:
            for k in range(min(len(got), len(ref))):
                gt, gs = got[k]
                if int(gt) == ref[k][0] and int(gs) == ref[k][1]:
                    correct += 1
        except Exception:
            pass

    return {"exact_match": (correct / total) if total else 0.0}
