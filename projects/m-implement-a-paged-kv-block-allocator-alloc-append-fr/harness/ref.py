from pagedkv.allocator import BlockAllocator, SequenceManager, compute_slot_mapping

def oracle_alloc_free(num_blocks, block_size, ops):
    alloc = BlockAllocator(num_blocks, block_size)
    history = []
    for op, val in ops:
        if op == "alloc":
            history.append(alloc.alloc())
        elif op == "free":
            alloc.free(val)
            history.append(alloc.get_ref_count(val))
    return history

def oracle_cow_and_slot(num_blocks, block_size):
    alloc = BlockAllocator(num_blocks, block_size)
    mgr = SequenceManager(alloc, block_size)
    mgr.create_sequence(1, 3)
    mgr.fork(1, 2)
    mgr.append_token(2)
    table1 = mgr.get_block_table(1)
    table2 = mgr.get_block_table(2)
    sm = compute_slot_mapping([table1, table2], [3, 4], block_size)
    return table1, table2, sm, alloc.ref_counts
