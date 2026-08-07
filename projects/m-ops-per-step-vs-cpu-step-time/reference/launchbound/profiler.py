"""Module for profiling step times and CPU launch overheads."""

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


def predict_small_batch_speedup(op_count, baseline_batch_size, target_batch_size, cpu_launch_overhead_us, gpu_time_per_op_per_batch_us):
    baseline_gpu_time = op_count * baseline_batch_size * gpu_time_per_op_per_batch_us
    baseline_cpu_time = op_count * cpu_launch_overhead_us
    baseline_step = max(baseline_cpu_time, baseline_gpu_time)

    target_gpu_time = op_count * target_batch_size * gpu_time_per_op_per_batch_us
    target_cpu_time = op_count * cpu_launch_overhead_us
    target_step = max(target_cpu_time, target_gpu_time)

    return baseline_step / target_step if target_step > 0 else 1.0
