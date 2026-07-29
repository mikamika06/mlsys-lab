import sys

sys.path.insert(0, ".")
from sched import Allocator


def test_allocate_release_roundtrip():
    a = Allocator(4, 16)
    assert a.free_count() == 4
    b = a.allocate()
    assert a.free_count() == 3
    a.release(b)
    assert a.free_count() == 4


def test_share_keeps_block_alive():
    a = Allocator(2, 16)
    b = a.allocate()
    a.share(b)
    a.release(b)
    assert a.free_count() == 1
    a.release(b)
    assert a.free_count() == 2
