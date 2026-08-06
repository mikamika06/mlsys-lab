import os
import torch
from torch.profiler import profile, schedule, ProfilerAction


def run_profiled_training(model, dataloader, optimizer, loss_fn, schedule_config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    traces_written = 0

    def trace_handler(p):
        nonlocal traces_written
        traces_written += 1
        file_path = os.path.join(output_dir, f"trace_{traces_written}.json")
        p.export_chrome_trace(file_path)

    prof_schedule = schedule(
        wait=schedule_config["wait"],
        warmup=schedule_config["warmup"],
        active=schedule_config["active"],
        repeat=schedule_config.get("repeat", 0)
    )

    with profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=prof_schedule,
        on_trace_ready=trace_handler
    ) as prof:
        for x, y in dataloader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            prof.step()

    return traces_written


def diagnose_step_invocations(trace_files_emitted: int, expected_active_cycles: int) -> dict:
    if expected_active_cycles == 0:
        is_double_step = trace_files_emitted > 0
    else:
        ratio = trace_files_emitted / expected_active_cycles
        is_double_step = ratio >= 1.8

    return {
        "double_step_detected": is_double_step,
        "emitted": trace_files_emitted,
        "expected": expected_active_cycles
    }
