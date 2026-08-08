import sys

sys.path.insert(0, ".")
from onednn.parser import parse_row
from onednn.analysis import analyze_log

def test_parse_valid_row():
    line = "onednn_verbose,info,exec,cpu,convolution,jit:avx512,g0,mb1,1.23ms,2.50ms"
    res = parse_row(line)
    assert res is not None
    assert res["primitive"] == "convolution"
    assert res["time_ms"] == 2.50

def test_analyze_reconciliation():
    lines = ["onednn_verbose,info,exec,cpu,convolution,jit:avx512,g0,mb1,1.23ms,2.50ms"]
    res = analyze_log(lines, 3.0)
    assert res["total_kernel_time_ms"] == 2.50
    assert "jit" in res["classes"]
