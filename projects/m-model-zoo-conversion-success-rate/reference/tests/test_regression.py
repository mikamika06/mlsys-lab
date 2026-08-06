import sys

sys.path.insert(0, ".")
from conv.classifier import classify_error
from conv.reader import read_signature
from conv.success import compute_success_rate


def test_classifier_unsupported_op():
    assert classify_error("OpCode not supported: CUSTOM_OP") == "unsupported_op"


def test_reader_signature():
    raw = b"SIG:default|INPUT:x:float32[1,10]|OUTPUT:y:float32[1,10]"
    res = read_signature(raw)
    assert res["signature"] == "default"
    assert "float32" in res["inputs"]


def test_success_rate_mixed():
    results = [{"success": True}, {"success": False}]
    assert compute_success_rate(results) == 0.5
