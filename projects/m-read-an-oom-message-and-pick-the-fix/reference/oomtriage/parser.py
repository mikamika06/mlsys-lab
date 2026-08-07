import re


def parse_oom_message(message: str) -> dict:
    result = {
        "requested_bytes": 0,
        "total_capacity_bytes": 0,
        "allocated_bytes": 0,
        "cached_free_bytes": 0,
        "fragmentation_indicated": False,
    }
    req_match = re.search(r"Tried to allocate ([\d.]+)\s*([KMG]iB)", message)
    if req_match:
        val = float(req_match.group(1))
        unit = req_match.group(2)
        mult = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}.get(unit, 1)
        result["requested_bytes"] = int(val * mult)
    cap_match = re.search(r"([\d.]+)\s*([KMG]iB) total capacity", message)
    if cap_match:
        val = float(cap_match.group(1))
        unit = cap_match.group(2)
        mult = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}.get(unit, 1)
        result["total_capacity_bytes"] = int(val * mult)
    alloc_match = re.search(r"([\d.]+)\s*([KMG]iB) already allocated", message)
    if alloc_match:
        val = float(alloc_match.group(1))
        unit = alloc_match.group(2)
        mult = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}.get(unit, 1)
        result["allocated_bytes"] = int(val * mult)
    cached_match = re.search(r"([\d.]+)\s*([KMG]iB) free in cached", message)
    if cached_match:
        val = float(cached_match.group(1))
        unit = cached_match.group(2)
        mult = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}.get(unit, 1)
        result["cached_free_bytes"] = int(val * mult)
    if "external fragmentation" in message.lower():
        result["fragmentation_indicated"] = True
    return result


def recommend_fix(parsed_info: dict) -> str:
    if parsed_info.get("fragmentation_indicated") or parsed_info.get("cached_free_bytes", 0) > parsed_info.get("requested_bytes", 0):
        return "tune_max_split_size"
    elif parsed_info.get("allocated_bytes", 0) + parsed_info.get("requested_bytes", 0) > parsed_info.get("total_capacity_bytes", 0):
        return "reduce_batch_size_or_gradient_checkpointing"
    else:
        return "clear_cache"
