def parse_transfer_logs(logs: list) -> bool:
    for line in logs:
        if "contention" in line or "suboptimal" in line or "small_chunk" in line:
            return True
    return False
