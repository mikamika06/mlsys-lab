import sys
sys.path.insert(0, ".")
from serverdiag.detector import classify_failure

def test_explicit_crash_is_crash():
    logs = ["[INFO] starting", "[ERROR] Segmentation fault"]
    metrics = [{"timestamp": 1, "cpu_util": 10.0, "gpu_util": 10.0, "alive": False}]
    assert classify_failure(logs, metrics) == "crash"

def test_silent_hang_is_hang():
    logs = ["[INFO] running", "[WARN] stalled"]
    metrics = [{"timestamp": 1, "cpu_util": 10.0, "gpu_util": 10.0, "alive": True}]
    assert classify_failure(logs, metrics) == "hang"
