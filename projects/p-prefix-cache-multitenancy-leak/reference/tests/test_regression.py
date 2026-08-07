import sys
sys.path.insert(0, ".")
from cache import PrefixCache
import ref

def test_isolation_and_hit_rate():
    alloc = ref.BlockAllocator()
    c = PrefixCache(4, alloc, isolation=True, shared_system=True)

    sys_toks = list(range(100, 140))

    c.insert(sys_toks, "A", is_system=True)

    assert len(c.match(sys_toks, "B")) == 10

    c.insert(sys_toks + [1, 2, 3, 4], "A", is_system=False)

    assert len(c.match(sys_toks + [1, 2, 3, 4], "B")) == 10
