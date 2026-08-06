def calculate_payback_volume(build_time_sec: float, base_latency_ms: float, target_latency_ms: float) -> int:
    raise NotImplementedError


def build_normalized_table(candidates: list[dict], baseline_backend: str = "ort_cuda") -> list[dict]:
    raise NotImplementedError
