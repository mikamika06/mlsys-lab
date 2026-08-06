import sys
sys.path.insert(0, ".")
from timer.profiler import verify_sync
from timer.regimes import classify_regimes

def test_verify_sync_catches_contamination():
    bad_profile = {"kernel_ms": 0.01, "host_ms": 5.0, "synced": False}
    try:
        verify_sync(bad_profile)
        assert False, "should have raised ValueError"
    except ValueError:
        pass

def test_classify_regimes_output_structure():
    profiles = [{"name": "test", "kernel_ms": 1.0, "host_ms": 1.0}]
    res = classify_regimes(profiles)
    assert isinstance(res, list)
    assert "regime_id" in res[0]
