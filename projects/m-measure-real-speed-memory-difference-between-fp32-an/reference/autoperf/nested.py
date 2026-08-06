import torch


def run_with_nested_disable(model, x):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        results.append(torch.is_autocast_enabled(device_type))

        with torch.autocast(device_type=device_type, enabled=False):
            results.append(torch.is_autocast_enabled(device_type))
            out = model(x)

        results.append(torch.is_autocast_enabled(device_type))

    return results, out
