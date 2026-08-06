import sys
sys.path.insert(0, ".")
import tempfile
from compcache.measurement import measure_compile_latencies
from compcache.invalidation import check_cache_behavior
from compcache.mega import run_and_save_cache, verify_zero_recompiles

def test_warm_compile_is_faster():
    """Ensure warm compile is faster or equal to cold."""
    with tempfile.TemporaryDirectory() as d:
        cold, warm = measure_compile_latencies(d)
        assert warm <= cold

def test_identical_run_hits_cache():
    """Ensure identical run hits cache."""
    with tempfile.TemporaryDirectory() as d:
        res = check_cache_behavior(d)
        assert res["identical_hit"] is True

def test_mega_cache_zero_recompiles():
    """Ensure mega cache yields zero recompiles."""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2, tempfile.TemporaryDirectory() as d3:
        run_and_save_cache(d1, d2)
        ok = verify_zero_recompiles(d2, d3)
        assert ok is True
