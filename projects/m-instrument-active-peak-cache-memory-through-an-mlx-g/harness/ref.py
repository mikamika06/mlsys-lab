def ref_instrument(steps, base_mem=1024):
    out = []
    active = base_mem
    cache = 0
    peak = base_mem
    for i in range(steps):
        active += 256
        cache += 128
        peak = max(peak, active + cache)
        out.append({"step": i, "active": active, "cache": cache, "peak": peak})
    return out

def ref_throughput(steps, memory_limit_enabled=False):
    base_time = steps * 0.01
    if memory_limit_enabled:
        return steps / (base_time * 1.8)
    return steps / base_time

def ref_locate(logs, ceiling):
    for line in logs:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            step = int(parts[0])
            mem = int(parts[1])
            if mem > ceiling:
                return step
    return -1

LOGS_SAMPLE = [
    "0 1024",
    "1 2048",
    "2 4096",
    "3 8192"
]
