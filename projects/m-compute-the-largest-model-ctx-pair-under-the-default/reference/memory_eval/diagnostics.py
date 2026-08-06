def detect_swap_thrash(tok_speeds, memory_pressures):
    if not tok_speeds or not memory_pressures:
        return False
    avg_speed = sum(tok_speeds) / len(tok_speeds)
    peak_pressure = max(memory_pressures)
    initial_speed = tok_speeds[0]
    if initial_speed > 0 and (avg_speed / initial_speed) < 0.2 and peak_pressure > 0.85:
        return True
    return False


def classify_failure(log_text):
    if "Metal alloc failed" in log_text or "out of memory during buffer creation" in log_text:
        return "metal_alloc_failure"
    if "Killed: 9" in log_text or "oom-killer" in log_text.lower() or "Out of memory: Kill process" in log_text:
        return "oom_kill"
    if "Runner exited with status" in log_text or "segmentation fault" in log_text.lower():
        return "runner_crash"
    return "unknown"
