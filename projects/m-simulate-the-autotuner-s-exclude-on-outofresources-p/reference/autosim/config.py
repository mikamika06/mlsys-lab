class OutOfResources(Exception):
    pass


def evaluate_config(config, workload_size):
    shmem = config.get("num_stages", 1) * config.get("block_m", 16) * config.get("block_n", 16) * 4
    regs = config.get("num_warps", 4) * 32
    if shmem > 49152 or regs > 256:
        raise OutOfResources("Exceeded hardware limits")
    return float(workload_size * 10 / (config.get("block_m", 16) * config.get("block_n", 16)))
