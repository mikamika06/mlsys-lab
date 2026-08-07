import sys
sys.path.insert(0, ".")
from pagedkv.cache import BlockCache

def test_blocks_are_released_properly():
    c = BlockCache(block_size=16, num_blocks=4, head_dim=32)
    b = c.allocate()
    c.release(b)
    assert c.free_count() == 4

def test_long_horizon_memory_stability():
    c = BlockCache(block_size=16, num_blocks=10, head_dim=32)
    allocated = []
    for _ in range(1000):
        if c.free_count() > 0:
            b = c.allocate()
            allocated.append(b)
        if len(allocated) > 5:
            old = allocated.pop(0)
            c.release(old)
    assert c.free_count() > 0
