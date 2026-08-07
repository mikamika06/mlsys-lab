"""Module for analyzing bounds and op pruning for hardware saturation."""

import math

def find_min_ops_for_busy_fraction(target_busy_fraction, cpu_launch_overhead_us, gpu_time_per_op_us):
    if target_busy_fraction <= 0.0:
        return 1
    if target_busy_fraction >= 1.0:
        if gpu_time_per_op_us >= cpu_launch_overhead_us:
            return 1
        return float("inf")

    needed_ops_denom = (1.0 - target_busy_fraction) * gpu_time_per_op_us
    if needed_ops_denom <= 0:
        return float("inf")

    req = (target_busy_fraction * cpu_launch_overhead_us) / (gpu_time_per_op_us * (1.0 - target_busy_fraction)) if gpu_time_per_op_us < cpu_launch_overhead_us else 1.0
    return max(1, math.ceil(req))


def cut_ops_until_busy_fraction(initial_ops, target_busy_fraction, cpu_launch_overhead_us, gpu_time_per_op_us):
    ops = initial_ops
    while ops > 0:
        cpu_time = ops * cpu_launch_overhead_us
        gpu_time = ops * gpu_time_per_op_us
        step_time = max(cpu_time, gpu_time)
        fraction = gpu_time / step_time if step_time > 0 else 0.0
        if fraction >= target_busy_fraction:
            return ops
        ops -= 1
    return 0
