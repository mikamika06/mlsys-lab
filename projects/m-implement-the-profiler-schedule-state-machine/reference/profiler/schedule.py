def get_step_action(step: int, skip_first: int, wait: int, warmup: int, active: int, repeat: int = 0) -> str:
    """Determine profiler action string for step."""
    if step < 0 or skip_first < 0 or wait < 0 or warmup < 0 or active < 0 or repeat < 0:
        raise ValueError("Step and schedule parameters must be non-negative.")
    if step < skip_first:
        return "NONE"
    s = step - skip_first
    cycle_len = wait + warmup + active
    if cycle_len == 0:
        return "NONE"
    cycle_idx = s // cycle_len
    if repeat > 0 and cycle_idx >= repeat:
        return "NONE"
    offset = s % cycle_len
    if offset < wait:
        return "NONE"
    if offset < wait + warmup:
        return "WARMUP"
    if offset < wait + warmup + active:
        if offset == wait + warmup + active - 1:
            return "RECORD_AND_SAVE"
        return "RECORD"
    return "NONE"


def schedule_summary(total_steps: int, skip_first: int, wait: int, warmup: int, active: int, repeat: int = 0) -> dict:
    """Compute schedule step breakdown and active ranges."""
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative.")
    none_c = 0
    warmup_c = 0
    record_c = 0
    save_c = 0
    active_ranges = []
    in_active = False
    active_start = 0

    for step in range(total_steps):
        act = get_step_action(step, skip_first, wait, warmup, active, repeat)
        if act == "NONE":
            none_c += 1
            if in_active:
                active_ranges.append((active_start, step))
                in_active = False
        elif act == "WARMUP":
            warmup_c += 1
            if in_active:
                active_ranges.append((active_start, step))
                in_active = False
        elif act == "RECORD":
            record_c += 1
            if not in_active:
                active_start = step
                in_active = True
        elif act == "RECORD_AND_SAVE":
            save_c += 1
            if not in_active:
                active_start = step
                in_active = True
            active_ranges.append((active_start, step + 1))
            in_active = False

    if in_active:
        active_ranges.append((active_start, total_steps))

    return {
        "total_steps": total_steps,
        "none_count": none_c,
        "warmup_count": warmup_c,
        "record_count": record_c,
        "record_and_save_count": save_c,
        "active_ranges": active_ranges,
    }
