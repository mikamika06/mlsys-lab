import json

def compute_memory_stats(log_lines):
    total_blocks = 0
    free_blocks = 0
    samples = 0

    for line in log_lines:
        if "GPU blocks" in line or "block_usage" in line or "free_kv_blocks" in line:
            try:
                data = json.loads(line.strip())
                if "total_blocks" in data and "free_blocks" in data:
                    total_blocks += data["total_blocks"]
                    free_blocks += data["free_blocks"]
                    samples += 1
            except Exception:
                pass

    if samples == 0:
        return {"avg_utilization": 0.0, "min_free_blocks": 0}

    avg_free = free_blocks / samples
    avg_total = total_blocks / samples
    utilization = 1.0 - (avg_free / avg_total) if avg_total > 0 else 0.0
    return {"avg_utilization": float(utilization), "min_free_blocks": int(avg_free)}
