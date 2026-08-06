def test_contiguous_block_matching():
    from caching.blocks import surviving_blocks

    new_prompt = [1, 2, 9, 9, 5, 6]
    cached_seqs = [[101, 102, 103]]
    block_contents = {
        101: [1, 2],
        102: [3, 4],
        103: [5, 6]
    }

    res = surviving_blocks(new_prompt, cached_seqs, block_contents)
    assert res == [101], f"Expected [101], got {res}"
