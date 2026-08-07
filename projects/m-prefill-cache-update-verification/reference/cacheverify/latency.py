def model_latencies(prefill_len, gen_steps, stateful_cost_per_token, stateless_cost_per_token):
    stateful_total = prefill_len * stateful_cost_per_token + gen_steps * stateful_cost_per_token
    stateless_total = prefill_len * stateless_cost_per_token + gen_steps * (prefill_len + gen_steps // 2) * stateless_cost_per_token
    return {
        "stateful_per_token": float(stateful_total / (prefill_len + gen_steps)),
        "stateless_per_token": float(stateless_total / (prefill_len + gen_steps))
    }
