def reconstruct_logical_kv(physical_store, block_table, block_size, num_valid_tokens):
    """Reconstruct contiguous logical KV from a paged physical store."""
    result = []
    for p in range(num_valid_tokens):
        logical_block = p // block_size
        slot = p % block_size
        phys_block = block_table[logical_block]
        result.append(list(physical_store[phys_block][slot]))
    return result
