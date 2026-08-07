import sys
sys.path.insert(0, ".")
from install.builder import execute_fallback

def test_fallback_behavior():
    res_low = execute_fallback({"compute_capability": 75})
    assert res_low["mode"] == "fallback"
    res_high = execute_fallback({"compute_capability": 89})
    assert res_high["mode"] == "native"
