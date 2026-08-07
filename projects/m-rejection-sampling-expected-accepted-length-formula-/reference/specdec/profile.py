import numpy as np

def speculative_speedup(short_prompt_metrics, long_prompt_metrics):
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
