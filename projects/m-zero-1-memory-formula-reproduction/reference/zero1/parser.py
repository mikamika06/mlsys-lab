import re


def parse_deepspeed_log(log_text):
    results = {}
    for line in log_text.splitlines():
        if "World size" in line or "world_size" in line.lower():
            match = re.search(r"(?:world_size|World size)[:\s]+(\d+)", line, re.IGNORECASE)
            if match:
                results["world_size"] = int(match.group(1))
        if "Optimizer" in line or "optimizer" in line.lower():
            match = re.search(r"(?:optimizer|Params)[:\s]+([\d,]+)", line, re.IGNORECASE)
            if match:
                val = int(match.group(1).replace(",", ""))
                results["num_params"] = val
    if "world_size" not in results:
        results["world_size"] = 1
    if "num_params" not in results:
        results["num_params"] = 0
    return results
