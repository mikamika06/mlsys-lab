def compute_kernel_intensity(flops: float, bytes_transferred: float) -> float:
    raise NotImplementedError


def aggregate_profile(records: list[dict]) -> dict[str, dict]:
    raise NotImplementedError


def model_total_stats(aggregated: dict[str, dict]) -> dict:
    raise NotImplementedError
