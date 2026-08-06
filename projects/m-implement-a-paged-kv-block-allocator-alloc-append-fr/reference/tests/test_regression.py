import sys
sys.path.insert(0, ".")
from pagedkv.allocator import BlockAllocator, SequenceManager, compute_slot_mapping

def test_allocator_basic():
    alloc = BlockAllocator(10, 4)
    b1 = alloc.alloc()
    b2 = alloc.alloc()
    assert alloc.get_ref_count(b1) == 1
    alloc.free(b1)
    assert alloc.get_ref_count(b1) == 0

def test_sequence_fork_cow():
    alloc = BlockAllocator(10, 4)
    mgr = SequenceManager(alloc, 4)
    mgr.create_sequence(1, 3)
    mgr.fork(1, 2)
    assert mgr.get_block_table(1) == mgr.get_block_table(2)
    assert alloc.get_ref_count(mgr.get_block_table(1)[0]) == 2
    mgr.append_token(2)
    assert mgr.get_block_table(1) != mgr.get_block_table(2)

def test_slot_mapping():
    tables = [[5, 2]]
    lens = [5]
    sm = compute_slot_mapping(tables, lens, 4)
    assert len(sm) == 5
    assert sm[0] == 5 * 4 + 0
    assert sm[4] == 2 * 4 + 0
