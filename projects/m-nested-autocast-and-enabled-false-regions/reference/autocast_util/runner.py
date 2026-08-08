import torch


def run_with_autocast(config, fn):
    results = []
    for item in config["nested"]:
        enabled = item["enabled"]
        dt_str = item["dtype"]
        dt = torch.float16 if dt_str == "float16" else (torch.bfloat16 if dt_str == "bfloat16" else torch.float32)
        with torch.cuda.amp.autocast(enabled=enabled, dtype=dt):
            results.append(fn())
    return results
