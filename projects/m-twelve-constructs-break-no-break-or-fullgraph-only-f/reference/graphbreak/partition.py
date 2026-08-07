def build_partition_map(log_lines):
    partitions = []
    current = []
    for line in log_lines:
        if "GRAPH BREAK" in line:
            if current:
                partitions.append(current)
                current = []
        else:
            clean = line.strip()
            if clean:
                current.append(clean)
    if current:
        partitions.append(current)
    return partitions
