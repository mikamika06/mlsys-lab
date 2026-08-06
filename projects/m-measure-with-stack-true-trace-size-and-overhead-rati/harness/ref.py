import random


def generate_traces(seed=42):
    random.seed(seed)
    traces_no_stack = [b"a" * random.randint(100, 500) for _ in range(5)]
    traces_with_stack = [b"a" * random.randint(300, 1200) for _ in range(5)]
    return traces_no_stack, traces_with_stack


def measure_trace_overhead(traces_no_stack, traces_with_stack):
    size_no = sum(len(t) for t in traces_no_stack)
    size_with = sum(len(t) for t in traces_with_stack)
    if size_no == 0:
        return 0.0
    return float(size_with) / float(size_no)


def active_windows_at_step(step: int, wait: int, warmup: int, active: int, repeat: int) -> int:
    if step < 0:
        return 0
    cycle_len = wait + warmup + active
    if cycle_len == 0:
        return 0
    count = 0
    for s in range(step + 1):
        c = s // cycle_len
        if repeat > 0 and c >= repeat:
            break
        pos = s % cycle_len
        if pos >= wait + warmup:
            count += 1
    return count
