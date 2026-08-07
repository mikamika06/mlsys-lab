import sys
sys.path.insert(0, ".")
from mpslab.cache import measure_reuse_overhead
from mpslab.graphs import compare_execution
from mpslab.trace import parse_trace


def test_parse_trace_basic():
    text = "Timestamp,Queue,BufferKind,Dur\n1,q,compute,10\n2,q,compute,20\n3,q,blit,15"
    res = parse_trace(text)
    assert res.get("compute") == 2
    assert res.get("blit") == 1


def test_compare_execution_modes():
    assert compare_execution("loop", 10) == 10
    assert compare_execution("graph", 10) == 1


def test_cache_overhead_inequality():
    cached_val = measure_reuse_overhead(True, 100)
    uncached_val = measure_reuse_overhead(False, 100)
    assert cached_val < uncached_val, "cached reuse should have lower overhead than uncached"
