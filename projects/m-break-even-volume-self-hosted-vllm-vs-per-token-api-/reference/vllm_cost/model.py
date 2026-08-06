def compute_breakeven_volume(fixed_monthly_cost, api_cost_per_token, self_hosted_variable_cost_per_token):
    if api_cost_per_token <= self_hosted_variable_cost_per_token:
        return float('inf')
    net_per_token = api_cost_per_token - self_hosted_variable_cost_per_token
    return fixed_monthly_cost / net_per_token


def compute_spot_expected_cost(base_hourly_cost, preemption_probability_per_hour, restart_overhead_hours, work_loss_fraction, requested_hours):
    effective_hourly_cost = base_hourly_cost * (1.0 + preemption_probability_per_hour * (restart_overhead_hours + work_loss_fraction))
    return effective_hourly_cost * requested_hours


def compute_prefix_caching_savings(total_requests, avg_prompt_tokens, avg_generation_tokens, cache_hit_rate, input_token_cost, cached_token_discount):
    total_prompt_tokens = total_requests * avg_prompt_tokens
    saved_tokens = total_prompt_tokens * cache_hit_rate
    savings_per_token = input_token_cost * cached_token_discount
    return saved_tokens * savings_per_token
