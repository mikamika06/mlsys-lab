import sys
sys.path.insert(0, ".")
from compcache.cache import CompilationCache
from compcache.invalidator import check_version

def test_version_invalidation():
    c = CompilationCache()
    c.set_version("v2")
    assert check_version(c, "v1") is False

def test_cache_no_stale_data():
    c = CompilationCache()
    c.store("k", b"val")
    assert c.lookup("k") == b"val"
    c.invalidate("k")
    assert c.lookup("k") is None
