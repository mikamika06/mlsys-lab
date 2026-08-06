def find_crossover(model_config, system_config):
    low = 1
    high = 20000
    best = high
    for b in range(low, high):
        bytes_per_block = b * system_config["block_size"] * model_config["kv_heads"] * model_config["head_dim"] * 2 * model_config["layers"]
        bandwidth = system_config["swap_bandwidth_gbps"] * 1e9 / 8
        sc = bytes_per_block / bandwidth
        tokens = b * system_config["block_size"]
        total_flops = tokens * system_config["flops_per_token"]
        rc = total_flops / 3e14
        if rc >= sc:
            best = b
            break
    return best
