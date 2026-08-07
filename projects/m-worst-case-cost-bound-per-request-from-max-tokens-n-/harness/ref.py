"""Reference data and oracle implementation for grading harness."""

REQUEST_CONFIGS = [
    {"prompt_tokens": 512, "max_tokens": 1024, "n": 1, "max_model_len": 4096},
    {"prompt_tokens": 3000, "max_tokens": 2000, "n": 4, "max_model_len": 4096},
    {"prompt_tokens": 100, "max_tokens": 50, "n": 2, "max_model_len": 2048},
    {"prompt_tokens": 4000, "max_tokens": 1000, "n": 1, "max_model_len": 4096},
    {"prompt_tokens": 2048, "max_tokens": 4096, "n": 8, "max_model_len": 4096},
]

PROFILE_PARAMS = {
    "prefill_sec_per_token": 0.0001,
    "decode_sec_per_token": 0.002,
    "base_overhead_sec": 0.05,
}


def oracle_calculate_worst_case_cost(request_config: dict, profile_params: dict) -> float:
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


def oracle_admit_request(request_config: dict, profile_params: dict, max_gpu_seconds: float) -> tuple[bool, float, str]:
    prompt_tokens = request_config.get("prompt_tokens", 0)
    max_model_len = request_config.get("max_model_len", 4096)

    if prompt_tokens > max_model_len:
        return False, 0.0, "EXCEEDS_MAX_MODEL_LEN"

    cost = oracle_calculate_worst_case_cost(request_config, profile_params)
    if cost > max_gpu_seconds:
        return False, cost, "COST_EXCEEDS_LIMIT"

    return True, cost, "ADMITTED"
