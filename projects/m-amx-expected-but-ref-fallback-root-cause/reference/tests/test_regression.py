import sys
sys.path.insert(0, ".")
from amxlog.parser import parse_verbose_line
from amxlog.sweep import analyze_k_sweep
from amxlog.profile import compute_time_dominance

def test_parser_valid_line():
    line = "onednn_verbose,info,cpu,convolution,jit:amx,forward,data:f32,g0_mb1_ic64_oc64_ih56_oh56_kh3_sh1_dh1_ph1_pw1,1.23"
    res = parse_verbose_line(line)
    assert res is not None
    assert res["primitive"] == "convolution"
    assert "amx" in res["jit"]

def test_sweep_selection():
    records = [{"k_val": 64, "kernel": "avx2"}, {"k_val": 1024, "kernel": "amx"}]
    res = analyze_k_sweep(records)
    assert res[64] == "avx2"
    assert res[1024] == "amx"

def test_profile_dominance():
    records = [{"primitive": "convolution", "time_ms": 80.0}, {"primitive": "pooling", "time_ms": 20.0}]
    res = compute_time_dominance(records)
    assert abs(res["convolution"] - 0.8) < 1e-5
