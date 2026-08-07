import sys
sys.path.insert(0, ".")
from flashres.resolver import parse_wheel_name
from flashres.compat import check_compatibility
from flashres.triage import triage_traceback

def test_parse_standard_wheel():
    w = "flash_attn-2.5.8-cp310-cp310-linux_x86_64.whl"
    res = parse_wheel_name(w)
    assert res["distribution"] == "flash_attn"
    assert res["version"] == "2.5.8"
    assert res["py_tag"] == "cp310"
    assert res["plat_tag"] == "linux_x86_64"

def test_compatibility_basic():
    record = {"distribution": "flash_attn", "version": "2.5.8", "py_tag": "cp310", "abi_tag": "cp310", "plat_tag": "linux_x86_64"}
    assert check_compatibility(record, "3.10", "12.1", "2.2.0") is True

def test_triage_recognition():
    tb = "PlatformError: flash_attn-2.5.8-cp39-cp39-win_amd64.whl is not a supported wheel on this platform."
    assert triage_traceback(tb) == "wheel_tag_mismatch"
