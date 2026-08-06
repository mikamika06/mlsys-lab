def compute_breakeven_volume(fixed_monthly_cost, api_cost_per_token, self_hosted_variable_cost_per_token):
    raise NotImplementedError


def compute_spot_expected_cost(base_hourly_cost, preemption_probability_per_hour, restart_overhead_hours, work_loss_fraction, requested_hours):
    raise NotImplementedError


def compute_prefix_caching_savings(total_requests, avg_prompt_tokens, avg_generation_tokens, cache_hit_rate, input_token_cost, cached_token_discount):
    raise NotImplementedError
