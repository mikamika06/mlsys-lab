def _exposed(partition, sizes, compute, bandwidth, latency):
    ready = []
    total = 0.0
    for t in compute:
        total += t
        ready.append(total)

    comm_end = 0.0
    for bucket in partition:
        start = max(max(ready[i] for i in bucket), comm_end)
        amount = sum(sizes[i] for i in bucket)
        comm_end = start + latency + amount / bandwidth

    return max(0.0, comm_end - total)


def bucket_grads(grad_sizes, compute_times, bandwidth, latency, max_bucket_bytes):
    n = len(grad_sizes)
    memo = {}

    def search(pos):
        if pos == n:
            return 0.0, []
        if pos in memo:
            return memo[pos]

        best = None
        amount = 0
        for end in range(pos, n):
            amount += grad_sizes[end]
            if amount > max_bucket_bytes:
                break
            rest_cost, rest = search(end + 1)
            part = [list(range(pos, end + 1))] + rest
            cost = _exposed(
                part,
                grad_sizes,
                compute_times,
                bandwidth,
                latency,
            )
            if best is None or cost < best[0]:
                best = (cost, part)

        memo[pos] = best
        return best

    return search(0)[1]
