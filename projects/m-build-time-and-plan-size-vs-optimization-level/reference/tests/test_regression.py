from trtpipe.diagnostics import classify_failure
from trtpipe.determinism import verify_roundtrip


def test_failure_classification():
    assert classify_failure("ONNX Parser error: unsupported operator") == "parser"
    assert classify_failure("Network definition has invalid layer dimension") == "network"
    assert classify_failure("BuilderConfig tactic selection failed: insufficient workspace") == "builder_config"
    assert classify_failure("Engine serialization failed during plan stream dump") == "engine"


def test_engine_determinism():
    valid_plan = b"TRT_PLAN_BINARY_DATA_V1"
    ok, digest = verify_roundtrip(valid_plan)
    assert ok is True
    assert isinstance(digest, str) and len(digest) == 64

    invalid_plan = b"BAD_HEADER_DATA"
    ok_bad, reason = verify_roundtrip(invalid_plan)
    assert ok_bad is False
    assert reason == "corrupted_header"
