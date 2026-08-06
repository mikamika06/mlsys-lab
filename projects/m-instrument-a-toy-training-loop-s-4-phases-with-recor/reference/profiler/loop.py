import torch


def profile_training_loop(step_fn, num_steps=5):
    """Run step_fn under profiler and report percentage time for 4 phases."""
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        for _ in range(num_steps):
            step_fn()

    phases = ["forward", "loss", "backward", "optimizer"]
    phase_times = {p: 0.0 for p in phases}

    for event in prof.function_events:
        if event.name in phase_times:
            phase_times[event.name] += event.cpu_time_total

    total_phase_time = sum(phase_times.values())
    if total_phase_time == 0:
        return {p: 0.0 for p in phases}

    return {p: (t / total_phase_time) * 100.0 for p, t in phase_times.items()}


def compute_uncovered_time_pct(step_fn, num_steps=5):
    """Compute percentage of total time spent outside record_function blocks."""
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        for _ in range(num_steps):
            step_fn()

    total_cpu_time = sum(evt.cpu_time_total for evt in prof.function_events)
    named_time = sum(
        evt.cpu_time_total
        for evt in prof.function_events
        if evt.name in ("forward", "loss", "backward", "optimizer")
    )

    if total_cpu_time == 0:
        return 0.0

    uncovered = max(0.0, total_cpu_time - named_time)
    return (uncovered / total_cpu_time) * 100.0
