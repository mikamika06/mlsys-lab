from trtplan.classifier import classify_engine


def test_hardware_penalty_calculated():
    """Verify hardware penalty calculation for cross-architecture compatible engines."""
    header = {
        "valid": True,
        "trt_version": (8, 6, 1, 0),
        "sm_arch": 80,
        "hardware_compatible": True,
        "platform": "linux-x86_64",
        "payload_size": 1024
    }
    env = {
        "trt_version": (8, 6, 1, 0),
        "sm_arch": 90,
        "platform": "linux-x86_64"
    }
    res = classify_engine(header, env)
    assert res["status"] == "OK"
    assert res["penalty"] is not None
    assert res["penalty"] > 1.0
