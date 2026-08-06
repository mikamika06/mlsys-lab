import random

def generate_scenarios(seed=42):
    rng = random.Random(seed)
    scenarios = []
    for _ in range(10):
        fixed = rng.uniform(500.0, 5000.0)
        api = rng.uniform(0.001, 0.005)
        self_var = api * rng.uniform(0.1, 0.5)
        
        base_cost = rng.uniform(1.0, 10.0)
        p_preempt = rng.uniform(0.01, 0.15)
        restart_h = rng.uniform(0.05, 0.3)
        loss_frac = rng.uniform(0.2, 0.8)
        req_h = rng.uniform(50.0, 500.0)
        
        tot_req = rng.randint(1000, 50000)
        avg_prompt = rng.randint(200, 2000)
        avg_gen = rng.randint(50, 500)
        hit_rate = rng.uniform(0.1, 0.9)
        input_cost = rng.uniform(0.000001, 0.00002)
        discount = rng.uniform(0.3, 0.8)
        
        scenarios.append({
            "breakeven": (fixed, api, self_var),
            "spot": (base_cost, p_preempt, restart_h, loss_frac, req_h),
            "cache": (tot_req, avg_prompt, avg_gen, hit_rate, input_cost, discount)
        })
    return scenarios

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
