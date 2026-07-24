import numpy as np


def _oracle(kernels, peak_flops, peak_bw):
    attainable = []
    for flops, bytes_moved in kernels:
        ai = flops / bytes_moved
        attainable.append(min(peak_flops, ai * peak_bw))
    order = sorted(range(len(kernels)), key=lambda i: (-attainable[i], i))
    return order


def _cases():
    rng = np.random.default_rng(0)
    cases = []

    # Hand-built: exact tie between two kernels that both hit peak.
    cases.append((
        [
            (1e9, 1e9),    # AI=1        -> bandwidth-bound
            (8e9, 1e6),    # AI=8000     -> compute-bound, hits peak (tie)
            (4e9, 1e7),    # AI=400      -> compute-bound, hits peak (tie)
            (2e8, 5e8),    # AI=0.4      -> bandwidth-bound, lowest
        ],
        2e12,
        4e11,
    ))

    # All bandwidth-bound, distinct AI -> ordering driven purely by AI.
    cases.append((
        [
            (3e8, 1e8),
            (1e8, 1e8),
            (5e8, 2e8),
            (2e8, 4e8),
        ],
        1e15,
        1e11,
    ))

    # Mixed regime, random.
    for _ in range(3):
        n = int(rng.integers(4, 9))
        flops = rng.uniform(1e6, 1e10, n)
        bytes_moved = rng.uniform(1e5, 1e9, n)
        peak_flops = float(rng.uniform(1e11, 1e13))
        peak_bw = float(rng.uniform(1e10, 1e12))
        cases.append((list(zip(flops.tolist(), bytes_moved.tolist())), peak_flops, peak_bw))

    return cases


def grade(sol, fx) -> dict:
    for kernels, peak_flops, peak_bw in _cases():
        expected = _oracle(kernels, peak_flops, peak_bw)
        try:
            got = sol.rank_kernels_by_throughput(list(kernels), peak_flops, peak_bw)
            got = [int(v) for v in got]
        except Exception:
            return {"exact_match": 0.0}

        if got != expected:
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
