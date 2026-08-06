import math


def simulate_hpa(load_trace, target_util, stabilization_window, min_replicas, max_replicas):
    replicas = min_replicas
    history = []
    scale_down_cooldown_left = 0

    for step, load in enumerate(load_trace):
        desired = math.ceil(replicas * (load / target_util)) if target_util > 0 else min_replicas
        desired = max(min_replicas, min(max_replicas, desired))

        if desired > replicas:
            replicas = desired
            scale_down_cooldown_left = stabilization_window
        elif desired < replicas:
            if scale_down_cooldown_left > 0:
                scale_down_cooldown_left -= 1
            else:
                replicas = desired
        else:
            if scale_down_cooldown_left > 0:
                scale_down_cooldown_left -= 1

        history.append(replicas)
    return history


def diagnose_thrash(timeline_events):
    for event in timeline_events:
        if event.get("oscillation_detected", False):
            return event.get("responsible_parameter", "stabilization_window")
    return "stabilization_window"
