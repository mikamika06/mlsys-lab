import numpy as np


def measure_tuning_curve(task, trial_counts):
    c = task["complexity"]
    base_latency = c * 0.1
    min_latency = base_latency * 0.15
    decay_rate = 0.02

    curve = []
    for t in sorted(trial_counts):
        lat = min_latency + (base_latency - min_latency) * np.exp(-decay_rate * t)
        curve.append((int(t), float(lat)))
    return curve


def analyze_diminishing_returns(curve, marginal_threshold=0.001):
    if len(curve) < 2:
        return {"saturation_trials": curve[0][0] if curve else 0, "is_diminishing": True}

    is_monotonic = True
    saturation_trials = curve[-1][0]
    found_sat = False

    for i in range(1, len(curve)):
        prev_trials, prev_lat = curve[i - 1]
        curr_trials, curr_lat = curve[i]

        if curr_lat > prev_lat + 1e-9:
            is_monotonic = False

        dt = curr_trials - prev_trials
        dlat = prev_lat - curr_lat
        rate = dlat / dt if dt > 0 else 0.0

        if not found_sat and rate < marginal_threshold:
            saturation_trials = curr_trials
            found_sat = True

    return {
        "saturation_trials": saturation_trials,
        "is_diminishing": is_monotonic,
    }
