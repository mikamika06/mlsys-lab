import torch


def run_profiled_training(model, dataloader, optimizer, loss_fn, schedule_config, output_dir):
    raise NotImplementedError


def diagnose_step_invocations(trace_files_emitted: int, expected_active_cycles: int) -> dict:
    raise NotImplementedError
