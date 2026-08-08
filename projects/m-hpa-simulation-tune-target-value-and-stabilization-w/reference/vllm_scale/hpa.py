import math


def simulate_hpa(load_trace, target_value, stabilization_window, min_replicas, max_replicas):
    current_replicas = min_replicas
    history = []
    desired_history = []
    for t, load in enumerate(load_trace):
        raw_desired = math.ceil(load / float(target_value))
        desired = max(min_replicas, min(max_replicas, raw_desired))
        desired_history.append(desired)
        window_start = max(0, t - stabilization_window + 1)
        window_desired = desired_history[window_start:t + 1]
        if desired > current_replicas:
            current_replicas = desired
        elif desired < current_replicas:
            current_replicas = max(window_desired)
        history.append(current_replicas)
    return history


def diagnose_thrash(load_trace, candidate_targets, candidate_windows, min_replicas, max_replicas):
    best_thrash = None
    best_idx = None
    best_param = None
    best_sim = None
    idx = 0
    for w in candidate_windows:
        for t in candidate_targets:
            sim = simulate_hpa(load_trace, t, w, min_replicas, max_replicas)
            flips = sum(1 for i in range(1, len(sim)) if sim[i] != sim[i - 1])
            if best_thrash is None or flips < best_thrash:
                best_thrash = flips
                best_idx = idx
                best_sim = sim
                if w == candidate_windows[0] and t != candidate_targets[0]:
                    best_param = "target_value"
                else:
                    best_param = "stabilization_window"
            idx += 1
    return {
        "argmin_index": best_idx,
        "responsible_parameter": best_param,
        "stabilized_trace": best_sim,
        "flips": best_thrash,
    }
