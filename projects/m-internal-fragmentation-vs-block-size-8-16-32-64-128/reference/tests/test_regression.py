import sys
sys.path.insert(0, ".")
from kvblocks.metrics import compute_fragmentation
from kvblocks.mapping import gather_slot_mapping
from kvblocks.trace import find_leaked_blocks

def test_gather_slot_mapping():
    seq_lens = [5]
    block_tables = [[10, 11]]
    block_size = 4
    want = [40, 41, 42, 43, 44]
    got = gather_slot_mapping(seq_lens, block_tables, block_size)
    assert got == want

def test_find_leaked_blocks():
    trace = [
        {"op": "alloc", "seq_id": 1, "blocks": [1, 2]},
        {"op": "free_block", "seq_id": 1, "block": 1},
        {"op": "free", "seq_id": 1}
    ]
    assert find_leaked_blocks(trace) == set()

    trace2 = [
        {"op": "alloc", "seq_id": 1, "blocks": [3]}
    ]
    assert find_leaked_blocks(trace2) == {3}

def test_fragmentation():
    assert compute_fragmentation([4], [4, 8]) == {4: 0, 8: 4}
