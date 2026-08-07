import sys
sys.path.insert(0, ".")
from trt_builder.compatibility import check_compatibility
from trt_builder.cache import enable_timing_cache
from trt_builder.profiles import lock_optimization_profiles
from trt_builder.inspector import inspect_diff, verify_tactics

def test_compatibility_valid():
    meta = {"cuda_version": 122, "tensorrt_version": 102, "gpu_compute_capability": 89}
    assert check_compatibility(meta) is True

def test_timing_cache_active():
    cfg = {}
    res = enable_timing_cache(cfg)
    assert res["timing_cache_enabled"] is True

def test_profiles_locked():
    p = [{"min_shape": [1, 3], "opt_shape": [4, 3], "max_shape": [8, 3]}]
    locked = lock_optimization_profiles(p)
    assert locked[0]["opt_shape"] == (4, 3)

def test_inspector_diff():
    ea = {"tactics": [1, 2, 3], "weights_hash": "abc"}
    eb = {"tactics": [1, 2, 3], "weights_hash": "xyz"}
    diffs = inspect_diff(ea, eb)
    assert "weights_hash" in diffs

def test_tactics_matching():
    ea = {"tactics": [1, 2, 3]}
    eb = {"tactics": [1, 2, 3]}
    assert verify_tactics(ea, eb) is True
