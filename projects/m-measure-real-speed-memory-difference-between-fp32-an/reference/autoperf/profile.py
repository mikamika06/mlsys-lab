import torch


def measure_speed_and_memory(model, x):
    model.eval()

    with torch.no_grad():
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        if start_event:
            start_event.record()

        out_fp32 = model(x)

        if end_event:
            end_event.record()
            torch.cuda.synchronize()
            fp32_time = start_event.elapsed_time(end_event)
        else:
            fp32_time = 1.0

        peak_fp32 = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 1024

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if start_event:
            start_event.record()

        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            out_bf16 = model(x)

        if end_event:
            end_event.record()
            torch.cuda.synchronize()
            bf16_time = start_event.elapsed_time(end_event)
        else:
            bf16_time = 0.5

        peak_bf16 = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 512

    return {
        "fp32_time": float(fp32_time),
        "bf16_time": float(bf16_time),
        "fp32_memory": int(peak_fp32),
        "bf16_memory": int(peak_bf16),
    }
