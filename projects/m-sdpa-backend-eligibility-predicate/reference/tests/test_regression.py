import sys

sys.path.insert(0, ".")
from sdpa_pred.eligibility import is_eligible
from sdpa_pred.predictor import predict_backend
from sdpa_pred.trace import detect_backend_from_trace


def test_math_always_eligible():
    assert is_eligible("math", "float32", True, 1024, 1024, 64, (7, 0)) is True


def test_flash_requires_ampere():
    assert is_eligible("flash_attention", "float16", False, 512, 512, 64, (7, 0)) is False
    assert is_eligible("flash_attention", "float16", False, 512, 512, 64, (8, 0)) is True


def test_predictor_returns_valid_backend():
    b = predict_backend("float16", True, 1024, 1024, 64, (8, 0))
    assert b in ("flash_attention", "mem_efficient", "math")


def test_trace_detection():
    events = [{"name": "flash_attn_fwd"}]
    assert detect_backend_from_trace(events) == "flash_attention"
