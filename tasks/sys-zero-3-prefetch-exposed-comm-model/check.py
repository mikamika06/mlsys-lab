def _oracle(layer_bytes, compute_times, bandwidth, prefetch_depth):
    exposed = 0.0
    n = len(layer_bytes)
    for i in range(n):
        overlap = 0.0
        start = max(0, i - prefetch_depth)
        for j in range(start, i):
            overlap += float(compute_times[j])
        comm_time = float(layer_bytes[i]) / float(bandwidth)
        hidden_time = min(comm_time, overlap)
        exposed += (comm_time - hidden_time) * float(bandwidth)
    return float(exposed)


def grade(sol, fx) -> dict:
    cases = [
        ([1000, 2000, 3000], [1.0, 0.5, 2.0], 1000, 1),
        ([4096, 4096, 4096, 4096], [0.1, 0.1, 0.1, 0.1], 2048, 2),
        ([800, 1200, 400, 900], [1.0, 1.0, 1.0, 1.0], 1000, 3),
        ([1_000_000, 500_000], [2.0, 0.5], 100_000, 0),
        ([700, 900, 1100, 1300, 1500], [0.4, 0.8, 0.2, 1.1, 0.3], 2500, 2),
    ]

    ok = 1.0
    for layer_bytes, compute_times, bandwidth, depth in cases:
        try:
            got = sol.exposed_comm_bytes(
                list(layer_bytes),
                list(compute_times),
                bandwidth,
                depth,
            )
        except Exception:
            ok = 0.0
            break
        ref = _oracle(layer_bytes, compute_times, bandwidth, depth)
        if abs(float(got) - ref) > 1e-9:
            ok = 0.0
            break

    return {"modeled_mem_access": ok}
