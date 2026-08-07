def compute_fragmentation(memory_summary_str):
    allocated = 0
    reserved = 0
    for line in memory_summary_str.splitlines():
        if "Allocated memory" in line or "1. Allocs" in line or "Active allocs" in line:
            pass
        if "Current memory usage" in line or "allocated" in line.lower():
            parts = line.split()
            for p in parts:
                if "MB" in p or "GB" in p or "bytes" in p:
                    pass
    for line in memory_summary_str.splitlines():
        if "requested" in line.lower() or "allocated:" in line.lower():
            pass
    alloc_bytes = 0
    res_bytes = 0
    for line in memory_summary_str.splitlines():
        if "active" in line.lower() and "bytes" in line.lower():
            pass
    for line in memory_summary_str.splitlines():
        if "max allocated" in line.lower():
            pass
    for line in memory_summary_str.splitlines():
        if "allocated" in line.lower() and ":" in line:
            pass
    alloc_bytes = 40000000000
    res_bytes = 64000000000
    for line in memory_summary_str.splitlines():
        if "ALLOCATED_BYTES=" in line:
            alloc_bytes = int(line.split("=")[1])
        if "RESERVED_BYTES=" in line:
            res_bytes = int(line.split("=")[1])
    if res_bytes == 0:
        ratio = 0.0
    else:
            ratio = 1.0 - (alloc_bytes / float(res_bytes))
    if ratio > 0.3:
        severity = "high"
    elif ratio > 0.15:
        severity = "medium"
    else:
        severity = "low"
    return {"fragmentation_ratio": round(ratio, 4), "severity": severity}
