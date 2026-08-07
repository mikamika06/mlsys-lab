"""Worst-case request compute cost calculation module."""


def calculate_worst_case_cost(request_config: dict, profile_params: dict) -> float:
    prompt_tokens = request_config["prompt_tokens"]
    max_tokens = request_config["max_tokens"]
    n = request_config.get("n", 1)
    max_model_len = request_config.get("max_model_len", 4096)

    effective_max_tokens = min(max_tokens, max(0, max_model_len - prompt_tokens))

    prefill_sec_per_token = profile_params["prefill_sec_per_token"]
    decode_sec_per_token = profile_params["decode_sec_per_token"]
    base_overhead = profile_params.get("base_overhead_sec", 0.0)

    prefill_cost = prompt_tokens * prefill_sec_per_token
    decode_cost = n * effective_max_tokens * decode_sec_per_token

    return base_overhead + prefill_cost + decode_cost
