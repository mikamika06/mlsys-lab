import torch


def parse_nested_regions(config):
    results = []
    current_enabled = False
    current_dtype = torch.float32
    for item in config["nested"]:
        enabled = item["enabled"]
        dt_str = item["dtype"]
        dt = torch.float16 if dt_str == "float16" else (torch.bfloat16 if dt_str == "bfloat16" else torch.float32)
        if not enabled:
            current_enabled = False
            current_dtype = torch.float32
        else:
            current_enabled = True
            current_dtype = dt
        results.append({"enabled": current_enabled, "dtype": str(current_dtype)})
    return results


def execute_in_nested_context(config, input_dtype):
    active_enabled = False
    active_dtype = torch.float32
    for item in config["nested"]:
        enabled = item["enabled"]
        dt_str = item["dtype"]
        dt = torch.float16 if dt_str == "float16" else (torch.bfloat16 if dt_str == "bfloat16" else torch.float32)
        if not enabled:
            active_enabled = False
            active_dtype = torch.float32
        else:
            active_enabled = True
            active_dtype = dt
    if not active_enabled:
        return torch.float32
    return active_dtype
