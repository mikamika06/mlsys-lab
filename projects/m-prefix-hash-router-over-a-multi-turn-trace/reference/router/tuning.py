from router.bakeoff import simulate_trace, compute_p95_ttft


def grid_search_alpha(
    requests: list[dict],
    num_workers: int,
    max_blocks_per_worker: int,
    block_size: int,
    prefill_rate: float,
    decode_rate: float,
    alphas: list[float] | None = None
) -> tuple[float, dict[float, float]]:
    if alphas is None:
        alphas = [round(i * 0.05, 2) for i in range(21)]

    p95_map = {}
    best_alpha = alphas[0]
    best_p95 = float("inf")

    for a in alphas:
        res = simulate_trace(
            requests,
            num_workers,
            max_blocks_per_worker,
            block_size,
            prefill_rate,
            decode_rate,
            policy="kv_aware",
            alpha=a
        )
        p95 = compute_p95_ttft(res)
        p95_map[a] = p95
        if p95 < best_p95:
            best_p95 = p95
            best_alpha = a

    return best_alpha, p95_map
