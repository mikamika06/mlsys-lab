import sys
sys.path.insert(0, ".")
from symfix.core import apply_check
from symfix.analysis import analyze_trace

def test_apply_check_includes_torch_check():
    cfg = {"input_dim": 16, "max_limit": 1024, "mode": "strict"}
    res = apply_check(cfg)
    assert "torch._check" in res, "apply_check must emit torch._check to avoid guard errors"
    assert "1024" in res

def test_analyze_trace_identifies_dynamic():
    t = {"trace": [10, 20, 30], "dynamic": True}
    assert analyze_trace(t) is True
