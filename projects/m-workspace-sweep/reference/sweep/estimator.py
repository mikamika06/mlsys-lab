def estimate_memory(config, profile_scale):
    total = 0
    for l in config["layers"]:
        total += l["base_mem"] + int(l["growth"] * profile_scale)
    return total
