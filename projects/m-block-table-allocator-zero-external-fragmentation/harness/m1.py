import ref


def check(workdir):
    from kvcache.allocator import BlockAllocator
    from kvcache.fragmentation import optimal_block_size

    seqs, _ = ref.get_fixtures()
    ref_best = ref.optimal_block_size(seqs, 128)
    got_best = optimal_block_size(seqs, 128)
    
    alloc = BlockAllocator(10)
    for _ in range(10):
        alloc.allocate()
    
    oom = False
    try:
        alloc.allocate()
    except MemoryError:
        oom = True
        
    return {
        "allocator_ooms_correctly": 1.0 if oom else 0.0,
        "optimal_size_matches": 1.0 if got_best == ref_best else 0.0,
    }
