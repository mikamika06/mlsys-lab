from paged_kv.allocator import BlockTableAllocator, compute_physical_blocks_needed


def test_block_allocator_fragmentation_invariant():
    allocator = BlockTableAllocator(num_blocks=10, block_size=16)
    tbl1 = allocator.allocate(seq_id=101, initial_seq_len=17)
    assert len(tbl1) == 2, "17 tokens with block_size 16 should require 2 blocks"

    metrics = compute_physical_blocks_needed([17], block_size=16)
    assert metrics["total_blocks"] == 2.0
    assert metrics["total_capacity"] == 32.0
    assert metrics["total_used_tokens"] == 17.0
    assert abs(metrics["fragmentation_ratio"] - (15.0 / 32.0)) < 1e-6

    allocator.append_tokens(seq_id=101, num_new_tokens=15)
    tbl1_after = allocator.get_block_table(seq_id=101)
    assert len(tbl1_after) == 2, "32 tokens with block_size 16 should still fit in 2 blocks"

    allocator.append_tokens(seq_id=101, num_new_tokens=1)
    tbl1_grow = allocator.get_block_table(seq_id=101)
    assert len(tbl1_grow) == 3, "33 tokens with block_size 16 must trigger 3rd block allocation"
