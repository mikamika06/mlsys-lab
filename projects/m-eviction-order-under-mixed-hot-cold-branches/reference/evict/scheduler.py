from evict.tree import eviction_order


def simulate_reclaim(config):
    order = eviction_order(config)
    cap = config["capacity"]
    total_nodes = len(config["nodes"])
    freed = max(0, total_nodes - cap)
    return order[:freed]
