def simulate_trace(
    requests: list[dict],
    num_workers: int,
    max_blocks_per_worker: int,
    block_size: int,
    prefill_rate: float,
    decode_rate: float,
    policy: str = "rr",
    alpha: float = 0.5
) -> list[dict]:
    raise NotImplementedError


def run_bakeoff(
    requests: list[dict],
    num_workers: int,
    max_blocks_per_worker: int,
    block_size: int,
    prefill_rate: float,
    decode_rate: float,
    alpha: float = 0.5
) -> dict[str, list[dict]]:
    raise NotImplementedError


def compute_p95_ttft(results: list[dict]) -> float:
    raise NotImplementedError
