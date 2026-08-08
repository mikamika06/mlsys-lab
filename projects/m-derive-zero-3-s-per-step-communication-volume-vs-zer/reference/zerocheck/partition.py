def extract_partition_sizes(init_log_lines, world_size):
    partitions = {}
    for line in init_log_lines:
        if "PARAM_INIT" in line:
            parts = line.strip().split()
            name = parts[1].split("=")[1]
            numel = int(parts[2].split("=")[1])
            base_size = numel // world_size
            rem = numel % world_size
            sizes = []
            for r in range(world_size):
                sizes.append(base_size + (1 if r < rem else 0))
            partitions[name] = sizes
    return partitions
