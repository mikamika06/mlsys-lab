import numpy as np

def reconstruct_logical_kv(physical_store, block_table, block_size, num_valid_tokens):
    """Reconstruct contiguous logical KV from a paged physical store."""
    bt = np.asarray(block_table, dtype=np.intp)
    positions = np.arange(num_valid_tokens, dtype=np.intp)
    logical_blocks = positions // block_size
    slots = positions % block_size
    phys = bt[logical_blocks]
    return physical_store[phys, slots]
