import re

def parse_memory_estimator_log(log_text):
    """
    Parses DeepSpeed ZeRO-2 memory estimator log text and extracts structured statistics.
    """
    res = {
        "params_numel": 0,
        "zero_stage": 0,
        "world_size": 1,
        "base_mem_gb": 0.0,
        "grad_mem_gb": 0.0,
        "total_mem_gb": 0.0
    }

    p_match = re.search(r"Total\s+Number\s+of\s+Parameters:\s*([\d_]+|[\d]+)", log_text, re.IGNORECASE)
    if p_match:
        res["params_numel"] = int(p_match.group(1).replace("_", ""))

    stage_match = re.search(r"ZeRO\s*Stage:\s*(\d+)", log_text, re.IGNORECASE)
    if stage_match:
        res["zero_stage"] = int(stage_match.group(1))

    ws_match = re.search(r"World\s*Size:\s*(\d+)", log_text, re.IGNORECASE)
    if ws_match:
        res["world_size"] = int(ws_match.group(1))

    base_match = re.search(r"Base\s+Memory:\s*([\d\.]+)\s*GB", log_text, re.IGNORECASE)
    if base_match:
        res["base_mem_gb"] = float(base_match.group(1))

    grad_match = re.search(r"Gradient\s+Memory:\s*([\d\.]+)\s*GB", log_text, re.IGNORECASE)
    if grad_match:
        res["grad_mem_gb"] = float(grad_match.group(1))

    tot_match = re.search(r"Total\s+Memory:\s*([\d\.]+)\s*GB", log_text, re.IGNORECASE)
    if tot_match:
        res["total_mem_gb"] = float(tot_match.group(1))

    return res
