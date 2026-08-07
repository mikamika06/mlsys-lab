import sys
sys.path.insert(0, ".")
from compcache.engine import CompilerEngine

def test_cache_hit():
    eng = CompilerEngine()
    _, c1 = eng.compile_and_run(1)
    _, c2 = eng.compile_and_run(1)
    assert c1 == 1
    assert c2 == 0

def test_warmup():
    eng = CompilerEngine()
    eng.warmup([[10]])
    _, c = eng.compile_and_run(10)
    assert c == 0

def test_invalidation():
    eng = CompilerEngine()
    eng.compile_and_run(5)
    eng.invalidate("v2")
    assert len(eng.cache) == 0
