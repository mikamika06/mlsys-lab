import os
import tempfile
import torch
from torch.profiler import profile, schedule, ProfilerAction

def run_training_loop(total_steps: int, schedule_params: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    trace_count = 0

    def trace_handler(prof):
        nonlocal trace_count
        trace_count += 1
        path = os.path.join(output_dir, f"trace_{trace_count}.json")
        prof.export_chrome_trace(path)

    prof_schedule = schedule(
        wait=schedule_params["wait"],
        warmup=schedule_params["warmup"],
        active=schedule_params["active"],
        repeat=schedule_params["repeat"]
    )

    with profile(
        schedule=prof_schedule,
        on_trace_ready=trace_handler,
        record_shapes=False,
        profile_memory=False,
        with_stack=False
    ) as prof:
        for _ in range(total_steps):
            prof.step()

    return trace_count

def diagnose_double_step(trace_files_count: int, schedule_params: dict, total_steps: int):
    from profiler_util.schedule import compute_actions
    actions = compute_actions(
        schedule_params["wait"],
        schedule_params["warmup"],
        schedule_params["active"],
        schedule_params["repeat"],
        total_steps
    )
    expected_saves = sum(1 for a in actions if a == ProfilerAction.RECORD_AND_SAVE)
    if trace_files_count > expected_saves:
        return "double_step_detected"
    return "normal"
