import sys
sys.path.insert(0, ".")
from exportcheck.sigdef import parse_signature_def
from exportcheck.classifier import classify_error
from exportcheck.success import compute_success_rate


def test_classifier_unsupported_op():
    assert classify_error("Error: OpCode not found for custom layer") == "unsupported_op"


def test_classifier_shape_mismatch():
    assert classify_error("Fatal: tensor shape mismatch between nodes") == "shape_mismatch"


def test_success_rate_calculation():
    records = [{"status": "success"}, {"status": "failed"}, {"status": "success"}]
    assert abs(compute_success_rate(records) - (2.0 / 3.0)) < 1e-5


def test_empty_records():
    assert compute_success_rate([]) == 0.0
