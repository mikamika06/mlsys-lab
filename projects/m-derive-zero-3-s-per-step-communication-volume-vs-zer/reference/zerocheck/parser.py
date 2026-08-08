def parse_memory_reduction(log_lines):
    results = {}
    for line in log_lines:
        if "MEM_REPORT" in line:
            parts = line.strip().split()
            rank = int(parts[1].split("=")[1])
            z1_mem = int(parts[2].split("=")[1])
            z3_mem = int(parts[3].split("=")[1])
            results[rank] = {"zero1": z1_mem, "zero3": z3_mem, "reduction_pct": (z1_mem - z3_mem) / z1_mem * 100.0}
    return results
