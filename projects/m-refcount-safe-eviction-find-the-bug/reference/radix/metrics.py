def tree_memory_overhead(node_count, pointer_size=8):
    node_struct_bytes = 64
    dict_overhead_bytes = 48
    return node_count * (node_struct_bytes + dict_overhead_bytes + pointer_size * 4)


def fork_reuse_savings(shared_tokens, total_tokens, bytes_per_token):
    return shared_tokens * bytes_per_token
