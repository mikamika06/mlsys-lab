import math

def derive_break_even(warmup_cost, eager_time, compiled_time):
    if compiled_time >= eager_time:
        return -1
    saved = eager_time - compiled_time
    return math.ceil(warmup_cost / saved)
