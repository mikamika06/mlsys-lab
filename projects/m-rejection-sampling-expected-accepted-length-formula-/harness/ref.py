import numpy as np

def generate_test_case(seed=42):
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.4, 0.95, size=5).tolist()

    short_trace = rng.binomial(1, 0.85, size=1000).tolist()
    long_trace = rng.binomial(1, 0.55, size=1000).tolist()
    trace_data = {"short_context": short_trace, "long_context": long_trace}

    short_m = {"target_time_per_token": 12.0, "draft_time_per_token": 1.5, "gamma": 4, "acceptance_rate": 0.85}
    long_m = {"target_time_per_token": 45.0, "draft_time_per_token": 3.0, "gamma": 4, "acceptance_rate": 0.55}

    return {
        "probs": probs,
        "trace_data": trace_data,
        "short_metrics": short_m,
        "long_metrics": long_m
    }

def expected_accepted_length_ref(probs):
    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return 0.0
    return float(np.sum(np.cumprod(p)))

def compute_acceptance_drop_ref(trace_data):
    s = np.asarray(trace_data["short_context"], dtype=np.float64)
    l = np.asarray(trace_data["long_context"], dtype=np.float64)
    rs = float(np.mean(s))
    rl = float(np.mean(l))
    return float((rs - rl) / rs)

def speculative_speedup_ref(short_prompt_metrics, long_prompt_metrics):
    t_target = float(short_prompt_metrics["target_time_per_token"])
    t_draft = float(short_prompt_metrics["draft_time_per_token"])
    gamma = int(short_prompt_metrics["gamma"])
    acc_short = float(short_prompt_metrics["acceptance_rate"])

    expected_accepted_short = (1.0 - acc_short**(gamma + 1)) / (1.0 - acc_short) - 1.0 if acc_short < 1.0 and acc_short > 0.0 else float(gamma)
    speedup_short = (1.0 + expected_accepted_short) / (1.0 + gamma * (t_draft / t_target))

    t_target_l = float(long_prompt_metrics["target_time_per_token"])
    t_draft_l = float(long_prompt_metrics["draft_time_per_token"])
    acc_long = float(long_prompt_metrics["acceptance_rate"])

    expected_accepted_long = (1.0 - acc_long**(gamma + 1)) / (1.0 - acc_long) - 1.0 if acc_long < 1.0 and acc_long > 0.0 else float(gamma)
    speedup_long = (1.0 + expected_accepted_long) / (1.0 + gamma * (t_draft_l / t_target_l))

    return {"speedup_short": float(speedup_short), "speedup_long": float(speedup_long)}
