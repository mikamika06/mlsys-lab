def identify_zero_hit(deployments):
    for d in deployments:
        if not d["enable_prefix_caching"] or d["cache_salt"] is None:
            return d["id"]
    return 3
