def _exposed(partition, sizes, compute, bandwidth, latency):
    ready = []
    total = 0.0
    for t in compute:
        total += t
        ready.append(total)

    comm_end = 0.0
    for bucket in partition:
        start_ready = max(ready[i] for i in bucket)
        bytes_in_bucket = sum(sizes[i] for i in bucket)
        start = max(start_ready, comm_end)
        comm_end = start + latency + bytes_in_bucket / bandwidth

    return max(0.0, comm_end - total)


def _oracle(sizes, compute, bandwidth, latency, cap):
    n = len(sizes)
    memo = {}

    def solve(pos):
        if pos == n:
            return (0.0, [])
        if pos in memo:
            return memo[pos]

        best = None
        used = 0
        for end in range(pos, n):
            used += sizes[end]
            if used > cap:
                break
            rest_cost, rest = solve(end + 1)
            candidate = ([list(range(pos, end + 1))] + rest)
            cost = _exposed(candidate, sizes, compute, bandwidth, latency)
            if best is None or cost < best[0]:
                best = (cost, candidate)
        memo[pos] = best
        return best

    return solve(0)


def grade(sol, fx) -> dict:
    cases = [
        ([40, 30, 50, 20], [1.0, 2.0, 1.0, 2.0], 100.0, 0.2, 80),
        ([10, 90, 10, 90, 10], [3.0, 1.0, 3.0, 1.0, 3.0], 120.0, 0.5, 100),
        ([25, 25, 25, 25, 25, 25], [1.0, 1.0, 5.0, 1.0, 1.0, 5.0], 200.0, 0.1, 75),
    ]

    worst = 1.0
    for sizes, compute, bandwidth, latency, cap in cases:
        try:
            got = sol.bucket_grads(
                list(sizes),
                list(compute),
                bandwidth,
                latency,
                cap,
            )
        except Exception:
            return {"size_ratio": float("inf")}

        valid = []
        for b in got:
            if not isinstance(b, list):
                return {"size_ratio": float("inf")}
            valid.extend(b)
            if sum(sizes[i] for i in b) > cap:
                return {"size_ratio": float("inf")}
            if b and b != list(range(b[0], b[-1] + 1)):
                return {"size_ratio": float("inf")}
        if sorted(valid) != list(range(len(sizes))):
            return {"size_ratio": float("inf")}

        optimal_cost, _ = _oracle(
            sizes, compute, bandwidth, latency, cap
        )
        candidate_cost = _exposed(
            got, sizes, compute, bandwidth, latency
        )

        if optimal_cost < 1e-12:
            ratio = 1.0 if candidate_cost < 1e-12 else float("inf")
        else:
            ratio = candidate_cost / optimal_cost
        worst = max(worst, ratio)

    return {"size_ratio": worst}
