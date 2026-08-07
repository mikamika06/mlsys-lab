import numpy as np

MOCK_RUNS = [
    {
        "accepted_lengths": [3, 4, 2, 5, 1, 4, 3, 3],
        "draft_times": [0.012, 0.015, 0.011, 0.016, 0.010, 0.014, 0.012, 0.013],
        "target_times": [0.025, 0.026, 0.024, 0.027, 0.024, 0.025, 0.025, 0.026],
        "verify_times": [0.003, 0.004, 0.003, 0.005, 0.002, 0.004, 0.003, 0.003],
        "overhead_times": [0.005, 0.006, 0.004, 0.007, 0.005, 0.005, 0.004, 0.005],
        "target_only_step_time": 0.025,
    },
    {
        "accepted_lengths": [1, 1, 2, 1, 0, 1],
        "draft_times": [0.020, 0.021, 0.019, 0.022, 0.020, 0.021],
        "target_times": [0.030, 0.031, 0.029, 0.032, 0.030, 0.031],
        "verify_times": [0.008, 0.007, 0.009, 0.008, 0.007, 0.008],
        "overhead_times": [0.010, 0.012, 0.011, 0.010, 0.009, 0.011],
        "target_only_step_time": 0.030,
    },
    {
        "accepted_lengths": [5, 5, 5, 4, 5],
        "draft_times": [0.005, 0.005, 0.005, 0.005, 0.005],
        "target_times": [0.040, 0.040, 0.040, 0.040, 0.040],
        "verify_times": [0.002, 0.002, 0.002, 0.002, 0.002],
        "overhead_times": [0.001, 0.001, 0.001, 0.001, 0.001],
        "target_only_step_time": 0.040,
    },
    {
        "accepted_lengths": [2, 3, 2, 4],
        "draft_times": [0.015, 0.016, 0.015, 0.017],
        "target_times": [0.035, 0.036, 0.035, 0.037],
        "verify_times": [0.005, 0.006, 0.005, 0.006],
        "overhead_times": [0.008, 0.009, 0.007, 0.008],
        "target_only_step_time": 0.035,
    },
]

MOCK_TRACE_EVENTS = [
    [
        {"name": "spec_step_0", "cat": "step", "dur": 45000, "ts": 0},
        {"name": "draft_model_forward", "cat": "draft", "dur": 12000, "ts": 100},
        {"name": "target_model_forward", "cat": "target", "dur": 25000, "ts": 12200},
        {"name": "sample_and_verify", "cat": "verify", "dur": 3000, "ts": 37300},
        {"name": "kv_cache_rollback", "cat": "overhead", "dur": 4800, "ts": 40300},
    ],
    [
        {"name": "spec_step_1", "cat": "step", "dur": 50000, "ts": 50000},
        {"name": "draft_model_forward", "cat": "draft", "dur": 15000, "ts": 50100},
        {"name": "target_model_forward", "cat": "target", "dur": 26000, "ts": 65200},
        {"name": "sample_and_verify", "cat": "verify", "dur": 4000, "ts": 91300},
        {"name": "kv_cache_rollback", "cat": "overhead", "dur": 4900, "ts": 95400},
    ],
]


def calculate_metrics(run_data):
    acc = np.array(run_data["accepted_lengths"], dtype=np.float64)
    draft = np.array(run_data["draft_times"], dtype=np.float64)
    target = np.array(run_data["target_times"], dtype=np.float64)
    verify = np.array(run_data["verify_times"], dtype=np.float64)
    overhead = np.array(run_data["overhead_times"], dtype=np.float64)
    t_target_step = float(run_data["target_only_step_time"])

    mean_accepted = float(np.mean(acc))
    total_tokens = float(np.sum(acc))
    total_spec_time = float(np.sum(draft + target + verify + overhead))

    real_tok_per_sec = total_tokens / total_spec_time
    baseline_tok_per_sec = 1.0 / t_target_step

    real_speedup = real_tok_per_sec / baseline_tok_per_sec
    theoretical_speedup = mean_accepted

    total_useful_time = float(np.sum(draft + target + verify))
    overhead_ratio = (total_spec_time - total_useful_time) / total_spec_time

    return {
        "mean_accepted_length": mean_accepted,
        "real_speedup": float(real_speedup),
        "theoretical_speedup": float(theoretical_speedup),
        "overhead_ratio": float(overhead_ratio),
    }


def parse_trace_events(events):
    cat_durations = {}
    total_dur = 0.0
    for ev in events:
        cat = ev.get("cat", "other")
        dur = float(ev.get("dur", 0.0))
        if cat != "step":
            cat_durations[cat] = cat_durations.get(cat, 0.0) + dur
            total_dur += dur

    if total_dur == 0.0:
        return {"draft": 0.0, "target": 0.0, "verify": 0.0, "overhead": 0.0}

    return {
        "draft": float(cat_durations.get("draft", 0.0) / total_dur),
        "target": float(cat_durations.get("target", 0.0) / total_dur),
        "verify": float(cat_durations.get("verify", 0.0) / total_dur),
        "overhead": float(cat_durations.get("overhead", 0.0) / total_dur),
    }
