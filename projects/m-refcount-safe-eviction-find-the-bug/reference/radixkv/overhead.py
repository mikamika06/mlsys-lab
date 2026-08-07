def tree_memory_overhead(node_count, branch_factor, metadata_bytes):
    base_node_size = metadata_bytes
    pointer_overhead = branch_factor * 8
    hash_overhead = 32
    total = node_count * (base_node_size + pointer_overhead + hash_overhead)
    return total
