import sys
sys.path.insert(0, ".")
from blk.analysis import internal_fragmentation, block_table_overhead, find_optimal_block_size, check_memory_threshold

def test_internal_frag():
    assert internal_fragmentation([10, 20], 16) == (6 + 12)

def test_table_overhead():
    assert block_table_overhead([16], 16, 8) == 8

def test_optimal_size():
    bs = [8, 16, 32]
    lengths = [15, 30, 45]
    opt = find_optimal_block_size(lengths, bs, 8, 1)
    assert opt in bs

def test_threshold():
    res = check_memory_threshold([16, 16], 16, 0.1)
    assert res is True
