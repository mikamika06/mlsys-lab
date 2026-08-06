def check(workdir):
    from paged_kv.allocator import BlockTableAllocator

    out = {"allocator_lifecycle_passed": 0.0}

    try:
        allocator = BlockTableAllocator(num_blocks=100, block_size=16)
        assert allocator.free_blocks_count == 100

        t1 = allocator.allocate(seq_id=1, initial_seq_len=20)
        assert len(t1) == 2
        assert allocator.free_blocks_count == 98

        t2 = allocator.allocate(seq_id=2, initial_seq_len=16)
        assert len(t2) == 1
        assert allocator.free_blocks_count == 97

        t1_append = allocator.append_tokens(seq_id=1, num_new_tokens=12)
        assert len(t1_append) == 2
        assert allocator.free_blocks_count == 97

        t1_append2 = allocator.append_tokens(seq_id=1, num_new_tokens=1)
        assert len(t1_append2) == 3
        assert allocator.free_blocks_count == 96

        allocator.free(seq_id=1)
        assert allocator.free_blocks_count == 99

        allocator.free(seq_id=2)
        assert allocator.free_blocks_count == 100

        out["allocator_lifecycle_passed"] = 1.0
    except Exception as e:
        out["_note"] = f"Allocator test failed: {type(e).__name__}: {e}"

    return out
