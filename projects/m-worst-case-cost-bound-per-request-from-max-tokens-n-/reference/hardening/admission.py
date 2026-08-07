"""Admission control based on bounded request costs."""

from hardening.cost import calculate_worst_case_cost


def admit_request(request_config: dict, profile_params: dict, max_gpu_seconds: float) -> tuple[bool, float, str]:
    prompt_tokens = request_config.get("prompt_tokens", 0)
    max_model_len = request_config.get("max_model_len", 4096)

    if prompt_tokens > max_model_len:
        return False, 0.0, "EXCEEDS_MAX_MODEL_LEN"

    cost = calculate_worst_case_cost(request_config, profile_params)
    if cost > max_gpu_seconds:
        return False, cost, "COST_EXCEEDS_LIMIT"

    return True, cost, "ADMITTED"
