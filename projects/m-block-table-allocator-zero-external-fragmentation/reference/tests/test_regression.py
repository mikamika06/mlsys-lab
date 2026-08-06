from kvcache.allocator import BlockAllocator


def test_zero_external_fragmentation():
    alloc = BlockAllocator(2)
    id0 = alloc.allocate()
    _ = alloc.allocate()
    alloc.free(id0)
    _ = alloc.allocate()
