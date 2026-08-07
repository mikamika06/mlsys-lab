import numpy as np


def calculate_metrics(run_data):
    """Calculates mean accepted length, wall-clock speedup, and overhead ratio."""
    acc = np.array(run_data["accepted_lengths"], dtype=np.float64)
    draft = np.array(run_data["draft_times"], dtype=np.float64)
    target = np.array(run_data["target_times"], dtype=np.float64)
    verify = np.array(run_data["verify_times"], dtype=np.float64)
    overhead = np.array(run_data["overhead_times"], dtype=np.float64)
    t_target_step = float(run_data["target_only_step_time"])

    mean_accepted = float(np.mean(acc))
    total_tokens = float(np.sum(acc))
    total_spec_time = float(np.sum(draft + target + verify + overhead))

    real_tok_per_sec = total_tokens / total_spec_time if total_spec_time > 0 else 0.0
    baseline_tok_per_sec = 1.0 / t_target_step if t_target_step > 0 else 0.0

    real_speedup = (
        real_tok_per_sec / baseline_tok_per_sec if baseline_tok_per_sec > 0 else 0.0
    )
    theoretical_speedup = mean_accepted

    total_useful_time = float(np.sum(draft + target + verify))
    overhead_ratio = (
        (total_spec_time - total_useful_time) / total_spec_time
        if total_spec_time > 0
        else 0.0
    )

    return {
        "mean_accepted_length": mean_accepted,
        "real_speedup": float(real_speedup),
        "theoretical_speedup": float(theoretical_speedup),
        "overhead_ratio": float(overhead_ratio),
    }
