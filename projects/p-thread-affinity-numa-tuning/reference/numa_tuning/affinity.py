def analyze_inter_node_traffic(node_topo, thread_allocations):
    remote_accesses = 0
    total_accesses = 0
    for thread, node in thread_allocations.items():
        total_accesses += 1
        if node_topo.get(thread) != node:
            remote_accesses += 1
    return float(remote_accesses) / float(total_accesses) if total_accesses > 0 else 0.0


def apply_pinning(thread_id, core_id):
    return {"thread": thread_id, "pinned_core": core_id, "status": "success"}


def allocate_numa_memory(size, node_id):
    return {"size": size, "node": node_id, "allocated": True}
