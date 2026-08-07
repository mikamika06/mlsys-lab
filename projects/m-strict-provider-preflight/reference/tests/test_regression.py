import sys

sys.path.insert(0, ".")
from ortpreflight.oracle import check_cuda_cudnn_compat
from ortpreflight.preflight import validate_preflight


def test_strict_preflight_fails_on_incompatible_cuda():
    env = {
        "ort_version": "1.18.0",
        "cuda_version": "11.0",
        "cudnn_version": "8.0",
        "device_count": 1,
    }
    res = validate_preflight(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        env,
        strict=True,
    )
    assert res["status"] == "FAILED", f"expected FAILED, got {res['status']}"
    assert res["selected_ep"] is None


def test_strict_preflight_fails_when_ep_missing():
    env = {
        "ort_version": "1.18.0",
        "cuda_version": "12.4",
        "cudnn_version": "9.1",
        "device_count": 1,
    }
    res = validate_preflight(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
        env,
        strict=True,
    )
    assert res["status"] == "FAILED", f"expected FAILED, got {res['status']}"


def test_non_strict_allows_fallback():
    env = {
        "ort_version": "1.18.0",
        "cuda_version": "11.0",
        "cudnn_version": "8.0",
        "device_count": 1,
    }
    res = validate_preflight(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
        env,
        strict=False,
    )
    assert res["status"] == "FALLBACK"
    assert res["selected_ep"] == "CPUExecutionProvider"
