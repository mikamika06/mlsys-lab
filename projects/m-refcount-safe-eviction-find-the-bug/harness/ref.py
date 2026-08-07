import random

CONFIGS = [
    {"node_count": 128, "branch_factor": 4, "metadata_bytes": 64},
    {"node_count": 512, "branch_factor": 2, "metadata_bytes": 64},
    {"node_count": 1024, "branch_factor": 8, "metadata_bytes": 128},
]

def simulate_eviction(tree_state, target_id):
    nodes = {n["id"]: dict(n) for n in tree_state["nodes"]}
    refcounts = {n["id"]: n["refcount"] for n in nodes.values()}
    parents = {n["id"]: n["parent"] for n in nodes.values()}

    current = target_id
    evicted = []
    while current is not None:
        if refcounts.get(current, 0) > 1:
            refcounts[current] -= 1
            break
        refcounts[current] = 0
        evicted.append(current)
        p = parents.get(current)
        if p is not None and p in nodes:
            if current in nodes[p].get("children", []):
                nodes[p]["children"].remove(current)
        current = p
    return sorted(evicted)

def tree_memory_overhead(node_count, branch_factor, metadata_bytes):
    base_node_size = metadata_bytes
    pointer_overhead = branch_factor * 8
    hash_overhead = 32
    total = node_count * (base_node_size + pointer_overhead + hash_overhead)
    return total

def fork_tokens_saved(base_tokens, num_branches, shared_prefix_len):
    saved_per_branch = shared_prefix_len
    total_saved = (num_branches - 1) * saved_per_branch
    return total_saved
