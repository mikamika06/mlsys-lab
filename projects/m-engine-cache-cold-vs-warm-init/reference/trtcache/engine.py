import numpy as np


def simulate_engine_init(config, is_cold):
    np.random.seed(config.get("seed", 42))
    base = config.get("base_latency", 10.0)
    if is_cold:
        overhead = config.get("compilation_overhead", 40.0)
        latency = base + overhead + float(np.random.rand() * 4.0)
        status = "compiled"
    else:
        latency = base + float(np.random.rand() * 1.0)
        status = "loaded_from_cache"
    return {"latency": latency, "status": status, "nodes_covered": config.get("total_nodes", 100)}
