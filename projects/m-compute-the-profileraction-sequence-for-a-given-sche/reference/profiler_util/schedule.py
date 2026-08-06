from torch.profiler import ProfilerAction

def compute_actions(wait: int, warmup: int, active: int, repeat: int, total_steps: int):
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
