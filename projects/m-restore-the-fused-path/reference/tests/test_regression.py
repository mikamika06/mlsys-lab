import sys

sys.path.insert(0, ".")
from flashfix.audit import triage_warnings
from flashfix.kernel import audit_contiguity
from flashfix.restore import restore_path

CONFIG = {"layer_id": 0, "shape": (16, 32, 128, 64), "strides": (262144, 8192, 64, 1), "contiguous": True}


def test_triage_recognizes_layout_warning():
    logs = ["WARNING: non-contiguous tensor layout detected; falling back to slow path"]
    res = triage_warnings(logs)
    assert res == ["layout"]


def test_audit_detects_contiguity():
    res = audit_contiguity([CONFIG])
    assert res[0]["contiguous"] is True


def test_restore_path_rejects_warnings():
    log = "WARNING: stride mismatch in key-value cache; disabling fused kernel"
    res = restore_path(CONFIG, log)
    assert res is False
