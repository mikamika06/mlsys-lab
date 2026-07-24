import math

import numpy as np


def _batch_formation_times(arrivals, preferred_size, max_queue_delay):
    order = sorted(range(len(arrivals)), key=lambda k: (arrivals[k], k))
    sorted_arrivals = [arrivals[k] for k in order]
    n = len(sorted_arrivals)

    formation = [0] * n
    i = 0
    queue = []
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

    out = [0] * n
    for pos, k in enumerate(order):
        out[k] = formation[pos]
    return out


def _p99(latencies):
    n = len(latencies)
    s = sorted(latencies)
    k = math.ceil(0.99 * n)
    k = max(1, min(n, k))
    return s[k - 1]


def _oracle(arrivals, candidate_caps, preferred_size, service_time, slo):
    best = None
    for cap in sorted(set(int(c) for c in candidate_caps)):
        formation = _batch_formation_times(arrivals, preferred_size, cap)
        latencies = [f - a + service_time for f, a in zip(formation, arrivals)]
        p99 = _p99(latencies)
        if p99 <= slo:
            best = cap
            break
    return best if best is not None else -1


def _synthetic_cases():
    rng = np.random.default_rng(67)
    cases = []
    for _ in range(6):
        n = int(rng.integers(6, 25))
        gaps = rng.integers(0, 6, size=n)
        arrivals = np.cumsum(gaps).tolist()
        preferred_size = int(rng.integers(2, 6))
        service_time = int(rng.integers(0, 6))
        candidate_caps = rng.choice(np.arange(1, 40), size=int(rng.integers(3, 8)), replace=False).tolist()
        slo = float(rng.integers(2, 40))
        cases.append((arrivals, candidate_caps, preferred_size, service_time, slo))
    return cases


def grade(sol, fx) -> dict:
    fixture_case = (
        fx["arrivals"].tolist(),
        [50, 30, 2, 20, 10, 15, 5],
        4,
        5,
        9.0,
    )
    cases = [fixture_case] + _synthetic_cases()

    total = 0
    correct = 0
    for arrivals, candidate_caps, preferred_size, service_time, slo in cases:
        ref = _oracle(arrivals, candidate_caps, preferred_size, service_time, slo)
        total += 1
        try:
            got = sol.min_cap_meeting_slo(list(arrivals), list(candidate_caps), preferred_size, service_time, slo)
            if int(got) == ref:
                correct += 1
        except Exception:
            pass

    return {"exact_match": (correct / total) if total else 0.0}
