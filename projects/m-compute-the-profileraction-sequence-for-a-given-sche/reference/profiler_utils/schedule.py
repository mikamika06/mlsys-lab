import torch
from torch.profiler import ProfilerAction


def compute_action_sequence(wait: int, warmup: int, active: int, repeat: int, total_steps: int) -> list:
    actions = []
    cycle_len = wait + warmup + active
    if cycle_len == 0:
        return [ProfilerAction.NONE] * total_steps

    for step in range(total_steps):
        if repeat > 0:
            cycle_idx = step // cycle_len
            if cycle_idx >= repeat:
                actions.append(ProfilerAction.NONE)
                continue
        step_in_cycle = step % cycle_len
        if step_in_cycle < wait:
            actions.append(ProfilerAction.NONE)
        elif step_in_cycle < wait + warmup:
            actions.append(ProfilerAction.WARMUP)
        elif step_in_cycle < wait + warmup + active:
            actions.append(ProfilerAction.RECORD_AND_SAVE)
        else:
            actions.append(ProfilerAction.NONE)
    return actions
