from allocator.simulator import CachingAllocator


def test_best_fit_vs_first_fit():
    allocator = CachingAllocator(segment_size=1000)
    h1 = allocator.malloc(500)
    h_sep1 = allocator.malloc(10)
    h2 = allocator.malloc(200)
    h_sep2 = allocator.malloc(10)
    allocator.free(h1)
    allocator.free(h2)
    h_new = allocator.malloc(150)
    block = allocator.handles[h_new]
    assert block.addr == 510


def test_coalescing():
    allocator = CachingAllocator(segment_size=1000)
    h1 = allocator.malloc(400)
    h2 = allocator.malloc(400)
    allocator.free(h1)
    allocator.free(h2)
    h3 = allocator.malloc(700)
    assert len(allocator.segments) == 1
