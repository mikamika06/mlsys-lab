def grid_search_alpha(
    requests: list[dict],
    num_workers: int,
    max_blocks_per_worker: int,
    block_size: int,
    prefill_rate: float,
    decode_rate: float,
    alphas: list[float] | None = None
) -> tuple[float, dict[float, float]]:
    raise NotImplementedError
