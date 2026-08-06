from torch.profiler import ProfilerAction

CONFIGS = [
    {"wait": 1, "warmup": 1, "active": 2, "repeat": 1, "total_steps": 6},
    {"wait": 0, "warmup": 2, "active": 2, "repeat": 2, "total_steps": 10},
    {"wait": 2, "warmup": 0, "active": 1, "repeat": 3, "total_steps": 12},
    {"wait": 1, "warmup": 1, "active": 1, "repeat": 0, "total_steps": 15},
    {"wait": 0, "warmup": 0, "active": 3, "repeat": 1, "total_steps": 5}
]

def compute_reference_actions(cfg):
    wait = cfg["wait"]
    warmup = cfg["warmup"]
    active = cfg["active"]
    repeat = cfg["repeat"]
    total_steps = cfg["total_steps"]

    actions = []
    cycle_length = wait + warmup + active
    for step in range(total_steps):
        if repeat > 0 and (step // cycle_length) >= repeat:
            actions.append(ProfilerAction.NONE)
            continue
        step_in_cycle = step % cycle_length
        if step_in_cycle < wait:
            actions.append(ProfilerAction.NONE)
        elif step_in_cycle < wait + warmup:
            actions.append(ProfilerAction.WARMUP)
        elif step_in_cycle < wait + warmup + active - 1:
            actions.append(ProfilerAction.RECORD)
        else:
            actions.append(ProfilerAction.RECORD_AND_SAVE)
    return actions
