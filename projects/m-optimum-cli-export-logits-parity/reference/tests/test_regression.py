import sys
sys.path.insert(0, ".")
from optimum_export.export_utils import verify_logits_parity, compare_export_metrics, validate_architecture, UnsupportedArchitectureError


def test_logits_parity_valid():
    assert verify_logits_parity([1.0, 2.0], [1.0001, 2.0002], 1e-3) is True


def test_logits_parity_invalid():
    assert verify_logits_parity([1.0, 2.0], [1.5, 2.5], 1e-3) is False


def test_compare_export_metrics_structure():
    res = compare_export_metrics("test_model")
    assert "time_ratio" in res
    assert res["time_ratio"] > 1.0


def test_validate_architecture_unsupported():
    try:
        validate_architecture("UnknownArch")
        assert False, "Should have raised UnsupportedArchitectureError"
    except UnsupportedArchitectureError:
        pass
