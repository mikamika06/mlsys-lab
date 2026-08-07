def analyze_nvme_logs(log_lines):
    read_bytes = 0
    write_bytes = 0
    read_bw_list = []
    write_bw_list = []
    hits = 0
    misses = 0
    stall_us = 0

    for line in log_lines:
        tokens = line.strip().split()
        if "[NVME]" not in tokens:
            continue
        idx = tokens.index("[NVME]")
        if idx + 1 >= len(tokens):
            continue
        op = tokens[idx + 1]

        kv = {}
        for item in tokens[idx + 2 :]:
            if "=" in item:
                k, v = item.split("=", 1)
                kv[k] = v

        if op == "READ":
            sz = int(kv.get("size_bytes", 0))
            dur = int(kv.get("duration_us", 1))
            read_bytes += sz
            read_bw_list.append((sz / 1e9) / (dur / 1e6))
        elif op == "WRITE":
            sz = int(kv.get("size_bytes", 0))
            dur = int(kv.get("duration_us", 1))
            write_bytes += sz
            write_bw_list.append((sz / 1e9) / (dur / 1e6))
        elif op == "PREFETCH":
            st = kv.get("status", "")
            if st == "HIT":
                hits += 1
            elif st == "MISS":
                misses += 1
        elif op == "STALL":
            dur = int(kv.get("duration_us", 0))
            stall_us += dur

    avg_read_bw = sum(read_bw_list) / len(read_bw_list) if read_bw_list else 0.0
    avg_write_bw = sum(write_bw_list) / len(write_bw_list) if write_bw_list else 0.0
    total_prefetch = hits + misses
    hit_rate = (hits / total_prefetch) if total_prefetch > 0 else 0.0

    return {
        "total_read_bytes": read_bytes,
        "total_write_bytes": write_bytes,
        "avg_read_bandwidth_gbps": round(avg_read_bw, 4),
        "avg_write_bandwidth_gbps": round(avg_write_bw, 4),
        "prefetch_hit_rate": round(hit_rate, 4),
        "total_stall_time_ms": round(stall_us / 1000.0, 4),
    }
