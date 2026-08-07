"""Reference oracle generation and verification helpers."""

import math

WORKLOAD_SPECS = [
    {
        "op_counts": [10, 50, 100, 500, 1000],
        "cpu_launch_overhead_us": 8.0,
        "gpu_time_per_op_us": 5.0,
    },
    {
        "op_counts": [20, 100, 200],
        "cpu_launch_overhead_us": 12.0,
        "gpu_time_per_op_us": 15.0,
    }
]

PRUNING_SPECS = [
    {
        "initial_ops": 100,
        "target_busy_fraction": 0.8,
        "cpu_launch_overhead_us": 10.0,
        "gpu_time_per_op_us": 25.0
    },
    {
        "initial_ops": 50,
        "target_busy_fraction": 0.8,
        "cpu_launch_overhead_us": 15.0,
        "gpu_time_per_op_us": 5.0
    }
]

SPEEDUP_SPECS = [
    {
        "op_count": 500,
        "baseline_batch_size": 64,
        "target_batch_size": 8,
        "cpu_launch_overhead_us": 10.0,
        "gpu_time_per_op_per_batch_us": 0.5
    },
    {
        "op_count": 200,
        "baseline_batch_size": 32,
        "target_batch_size": 1,
        "cpu_launch_overhead_us": 20.0,
        "gpu_time_per_op_per_batch_us": 1.0
    }
]


def analyze_step(op_counts, cpu_launch_overhead_us, gpu_time_per_op_us):
    results = []
    for ops in op_counts:
        total_cpu_time = ops * cpu_launch_overhead_us
        total_gpu_time = ops * gpu_time_per_op_us
        step_time = max(total_cpu_time, total_gpu_time)
        busy_fraction = total_gpu_time / step_time if step_time > 0 else 0.0
        results.append({
            "ops": ops,
            "cpu_time_us": total_cpu_time,
            "gpu_time_us": total_gpu_time,
            "step_time_us": step_time,
            "gpu_busy_fraction": busy_fraction,
            "is_launch_bound": total_cpu_time > total_gpu_time
        })
    return results


def find_min_ops_for_busy_fraction(target_busy_fraction, cpu_launch_overhead_us, gpu_time_per_op_us):
    if target_busy_fraction <= 0.0:
        return 1
    if target_busy_fraction >= 1.0:
        if gpu_time_per_op_us >= cpu_launch_overhead_us:
            return 1
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


def predict_small_batch_speedup(op_count, baseline_batch_size, target_batch_size, cpu_launch_overhead_us, gpu_time_per_op_per_batch_us):
    baseline_gpu_time = op_count * baseline_batch_size * gpu_time_per_op_per_batch_us
    baseline_cpu_time = op_count * cpu_launch_overhead_us
    baseline_step = max(baseline_cpu_time, baseline_gpu_time)

    target_gpu_time = op_count * target_batch_size * gpu_time_per_op_per_batch_us
    target_cpu_time = op_count * cpu_launch_overhead_us
    target_step = max(target_cpu_time, target_gpu_time)

    return baseline_step / target_step if target_step > 0 else 1.0
