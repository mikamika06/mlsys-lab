import sys
sys.path.insert(0, ".")
from kvslot.unified import compare_unified_vs_perslot

def test_unified_vs_perslot_efficiency():
    res = compare_unified_vs_perslot(4, 8589934592, 1048576)
    assert isinstance(res, dict)
    assert "efficient" in res
    assert res["perslot_vram"] > 0
