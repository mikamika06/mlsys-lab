import numpy as np

from mlsys import scorers


def _simulate(arrivals, P, cap, D, T):
    """Real oracle: single-server dynamic-batcher event simulation.

    Each cycle, once the server is free and at least one request is
    queued, it dispatches as soon as EITHER trigger fires (whichever
    comes first):
      - size trigger: the queue already holds >= P requests,
      - delay trigger: the oldest queued request has waited D seconds.
    The dispatched batch takes min(queue length at trigger time, cap)
    requests (FIFO) and takes a fixed T seconds to finish, during which
    the server is unavailable for the next batch.
    """
    arrivals = np.asarray(arrivals, dtype=np.float64)
    n = len(arrivals)
    pos = 0
    server_free = 0.0
    lat_sum = 0.0
    count = 0
    last_finish = 0.0

    while pos < n:
        start_wait = max(server_free, arrivals[pos])
        j = pos
        while j < n and arrivals[j] <= start_wait:
            j += 1
        qcount = j - pos

        if qcount >= P:
            dispatch_time = start_wait
            take = min(qcount, cap)
        else:
            deadline = arrivals[pos] + D
            if pos + P - 1 < n and arrivals[pos + P - 1] <= deadline:
                dispatch_time = max(arrivals[pos + P - 1], start_wait)
                take = P
            else:
                dispatch_time = deadline
                k = pos
                while k < n and arrivals[k] <= deadline:
                    k += 1
                take = min(k - pos, cap)

        batch_end = pos + take
        finish_time = dispatch_time + T
        lat_sum += float(np.sum(finish_time - arrivals[pos:batch_end]))
        count += take
        last_finish = finish_time
        server_free = finish_time
        pos = batch_end

    mean_latency = lat_sum / count
    makespan = last_finish - arrivals[0]
    throughput = count / makespan
    return mean_latency, throughput


def _oracle(arrivals, cap, D, T, P_a, P_b):
    ml_a, tp_a = _simulate(arrivals, P_a, cap, D, T)
    ml_b, tp_b = _simulate(arrivals, P_b, cap, D, T)
    return np.array([ml_a, tp_a, ml_b, tp_b])


def _cases():
    cases = []
    # Primary fixture-driven comparison is added in grade() using fx.

    # A couple of small, hand-built synthetic cases for extra coverage.
    rng = np.random.default_rng(7)
    arrivals = np.cumsum(rng.exponential(1.0 / 200.0, size=300))
    cases.append((arrivals, 16, 0.03, 0.005, 2, 12))

    arrivals2 = np.cumsum(rng.exponential(1.0 / 50.0, size=150))
    cases.append((arrivals2, 8, 0.1, 0.02, 3, 8))

    return cases


def grade(sol, fx) -> dict:
    arrivals = fx["arrivals"]
    cap, D, T, P_a, P_b = 32, 0.05, 0.01, 4, 24

    all_cases = [(arrivals, cap, D, T, P_a, P_b)] + _cases()

    worst = 0.0
    for arrivals_c, cap_c, D_c, T_c, Pa_c, Pb_c in all_cases:
        ref = _oracle(arrivals_c, cap_c, D_c, T_c, Pa_c, Pb_c)
        try:
            got = sol.compare_preferred_batch_sizes(
                arrivals_c.copy(), cap_c, D_c, T_c, Pa_c, Pb_c
            )
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"rel_err": float("inf")}

        err = scorers.rel_err(ref, got)
        worst = max(worst, err)

    return {"rel_err": worst}
