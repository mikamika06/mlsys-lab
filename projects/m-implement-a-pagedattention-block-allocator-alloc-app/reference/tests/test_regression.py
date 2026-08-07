import sys

sys.path.insert(0, ".")
from kv.allocator import PagedAllocator

def test_cow_isolation():
    alloc = PagedAllocator(10, 4)
    alloc.alloc("s1", [1, 2])
    alloc.fork("s1", "s2")
    alloc.append("s2", [3])

    assert alloc.reconstruct("s1") == [1, 2], "Parent sequence corrupted!"
    assert alloc.reconstruct("s2") == [1, 2, 3], "Child sequence did not append correctly!"
