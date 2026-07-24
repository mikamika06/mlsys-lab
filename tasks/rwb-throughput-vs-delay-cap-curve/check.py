import numpy as np

from mlsys import scorers


def _batch_sizes(arrivals, preferred_size, max_queue_delay):
    arrivals = sorted(int(a) for a in arrivals)
    n = len(arrivals)
    i = 0
    queue = []
    sizes = []

    while i < n or queue:
        next_arrival = arrivals[i] if i < n else None
        next_timeout = queue[0] + max_queue_delay if queue else None

        if next_arrival is not None and (next_timeout is None or next_arrival <= next_timeout):
            t = next_arrival
            while i < n and arrivals[i] == t:
                queue.append(arrivals[i])
                i += 1
                if len(queue) == preferred_size:
                    sizes.append(preferred_size)
                    queue = []
        else:
            sizes.append(len(queue))
            queue = []

    return sizes


def _oracle(arrivals, caps, preferred_size, service_time):
    n = len(arrivals)
    mean_batch_size = []
    throughput = []
    for cap in caps:
        sizes = _batch_sizes(arrivals, preferred_size, int(cap))
        num_batches = len(sizes)
        mbs = (n / num_batches) if num_batches else 0.0
        mean_batch_size.append(mbs)
        throughput.append(mbs / service_time)
    return mean_batch_size, throughput


def _synthetic_cases():
    rng = np.random.default_rng(101)
    cases = []
    for _ in range(4):
        n = int(rng.integers(6, 25))
        gaps = rng.integers(0, 6, size=n)
        arrivals = np.cumsum(gaps).tolist()
        preferred_size = int(rng.integers(2, 6))
        service_time = float(rng.uniform(0.5, 4.0))
        caps = rng.choice(np.arange(1, 30), size=int(rng.integers(2, 6)), replace=False).tolist()
        cases.append((arrivals, caps, preferred_size, service_time))
    return cases


def grade(sol, fx) -> dict:
    fixture_case = (fx["arrivals"].tolist(), [3, 8, 15, 30, 50], 3, 2.0)
    cases = [fixture_case] + _synthetic_cases()

    worst = 0.0
    for arrivals, caps, preferred_size, service_time in cases:
        ref_mbs, ref_thr = _oracle(arrivals, caps, preferred_size, service_time)
        try:
            got = sol.throughput_vs_cap_curve(list(arrivals), list(caps), preferred_size, service_time)
            got_mbs = [float(x) for x in got["mean_batch_size"]]
            got_thr = [float(x) for x in got["throughput"]]
        except Exception:
            return {"rel_err": float("inf")}

        if len(got_mbs) != len(ref_mbs) or len(got_thr) != len(ref_thr):
            return {"rel_err": float("inf")}

        ref_vec = np.array(ref_mbs + ref_thr)
        got_vec = np.array(got_mbs + got_thr)
        err = scorers.rel_err(ref_vec, got_vec)
        worst = max(worst, err)

    return {"rel_err": worst}
