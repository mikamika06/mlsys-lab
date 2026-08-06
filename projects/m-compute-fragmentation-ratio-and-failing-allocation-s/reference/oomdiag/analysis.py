def compute_fragmentation_and_failing_size(dump_str):
    lines = dump_str.strip().split("\n")
    free_blocks = []
    req = 0
    for line in lines:
        parts = line.split(":")
        if len(parts) == 2:
            key, val = parts[0].strip(), int(parts[1].strip())
            if key == "free_block":
                free_blocks.append(val)
            elif key == "requested_allocation":
                req = val
    total_free = sum(free_blocks)
    max_free = max(free_blocks) if free_blocks else 0
    frag = 1.0 - (max_free / total_free) if total_free > 0 else 0.0
    return float(frag), int(req)
