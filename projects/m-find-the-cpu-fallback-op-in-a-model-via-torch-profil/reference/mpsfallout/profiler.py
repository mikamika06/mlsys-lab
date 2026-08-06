import torch


def find_fallback_ops(model, sample_input):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_dev = model.to(device)
    x = sample_input.to(device)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.MPS if hasattr(torch.profiler.ProfilerActivity, "MPS") else torch.profiler.ProfilerActivity.CPU
        ],
        record_shapes=True,
        profile_memory=False
    ) as prof:
        with torch.no_grad():
            _ = model_dev(x)
    events = prof.key_averages()
    fallback_ops = []
    for evt in events:
        if "copy_" in evt.key or "to" in evt.key or "CPU" in evt.key:
            if evt.key not in fallback_ops:
                fallback_ops.append(evt.key)
    return fallback_ops
