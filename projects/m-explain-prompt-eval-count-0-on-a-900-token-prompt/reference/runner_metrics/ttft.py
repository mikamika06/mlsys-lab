def compute_ttft(context_len: int, params: dict) -> float:
    prefill_time = context_len * params.get("time_per_prefill_token", 0.001)
    overhead = params.get("fixed_overhead", 0.02)
    return prefill_time + overhead


def dominant_component(context_len: int, params: dict) -> str:
    prefill_time = context_len * params.get("time_per_prefill_token", 0.001)
    overhead = params.get("fixed_overhead", 0.02)
    if prefill_time >= overhead:
        return "prefill"
    return "overhead"
