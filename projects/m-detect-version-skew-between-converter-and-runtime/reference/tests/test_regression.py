import sys
sys.path.insert(0, ".")
from skew.checker import detect_skew
from skew.runtime import get_supported_versions

def test_compatible_metadata():
    meta = {"version": 2, "alignment": 32}
    caps = get_supported_versions()
    res = detect_skew(meta, caps)
    assert res["status"] == "compatible"

def test_incompatible_version():
    meta = {"version": 99, "alignment": 32}
    caps = get_supported_versions()
    res = detect_skew(meta, caps)
    assert res["status"] == "incompatible"

def test_unsupported_alignment():
    meta = {"version": 2, "alignment": 128}
    caps = get_supported_versions()
    res = detect_skew(meta, caps)
    assert res["status"] == "incompatible"
