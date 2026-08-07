import sys
sys.path.insert(0, ".")
from zeroperf.parser import parse_log
from zeroperf.metrics import compute_overhead, compute_rel_err
from zeroperf.stats import summary_stats
import ref

def test_parse_log_extracts_correct_values():
    log = "INFO:step_time: 1.23\nINFO:step_time: 1.45"
    times = parse_log(log)
    assert times == [1.23, 1.45]

def test_zero3_overhead_is_positive():
    z2, z3 = ref.LOGS
    oh = compute_overhead(z2, z3, warmup=10)
    assert oh > 0.0

def test_rel_err_bounds():
    err = compute_rel_err(1.05, 1.00)
    assert 0.0 <= err <= 0.1

def test_summary_stats_keys():
    z2, _ = ref.LOGS
    st = summary_stats(z2, warmup=10)
    assert "mean" in st
    assert "median" in st
    assert "p95" in st
