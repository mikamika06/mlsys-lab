import re


def device_memory_report(engine_report_str):
    match = re.search(r"total_device_memory:\s*(\d+)", engine_report_str)
    total = int(match.group(1)) if match else 0
    match_free = re.search(r"free_device_memory:\s*(\d+)", engine_report_str)
    free = int(match_free.group(1)) if match_free else 0
    return {"total_device_memory": total, "free_device_memory": free}
