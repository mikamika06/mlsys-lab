import sys
sys.path.insert(0, ".")
from profops.diagnose import detect_sync_hotspot

def test_detect_sync_hotspot_flags_aten_copy():
    rows = [
        {"Name": "aten::copy_", "Self CPU total": "100.0ms", "Calls": 5000}
    ]
    res = detect_sync_hotspot(rows)
    assert res == "aten::copy_", f"expected aten::copy_, got {res}"

def test_detect_sync_hotspot_normal_op():
    rows = [
        {"Name": "aten::matmul", "Self CPU total": "10.0ms", "Calls": 10}
    ]
    res = detect_sync_hotspot(rows)
    assert res != "aten::copy_"
