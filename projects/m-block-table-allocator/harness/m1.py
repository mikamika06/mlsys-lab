def check(workdir):
    try:
        from pagedkv.allocator import BlockAllocator
    except Exception as e:
        return {"allocator_matched": 0.0, "_note": f"Import failed: {e}"}

    allocator = BlockAllocator(num_blocks=16, block_size=16)
    try:
        t1 = allocator.allocate("s1", 30)
        if len(t1) != 2:
            return {"allocator_matched": 0.0, "_note": f"Expected 2 blocks for 30 tokens, got {len(t1)}"}

        t1_updated = allocator.append_slots("s1", 35)
        if len(t1_updated) != 3:
            return {"allocator_matched": 0.0, "_note": f"Expected 3 blocks for 35 tokens, got {len(t1_updated)}"}

        t2 = allocator.allocate("s2", 10)
        if len(t2) != 1:
            return {"allocator_matched": 0.0, "_note": f"Expected 1 block for 10 tokens, got {len(t2)}"}

        if allocator.get_num_free_blocks() != 12:
            return {"allocator_matched": 0.0, "_note": f"Expected 12 free blocks, got {allocator.get_num_free_blocks()}"}

        allocator.free("s1")
        if allocator.get_num_free_blocks() != 15:
            return {"allocator_matched": 0.0, "_note": f"Expected 15 free blocks after free, got {allocator.get_num_free_blocks()}"}

        if allocator.get_block_table("s1") != []:
            return {"allocator_matched": 0.0, "_note": "Freed sequence block table should be empty"}
    except Exception as e:
        return {"allocator_matched": 0.0, "_note": f"Runtime error: {e}"}

    return {"allocator_matched": 1.0}
