import sys
sys.path.insert(0, ".")
from dnnlog import parse_row, classify_impl, reconcile_timing

LOG_LINES = [
    "oneDNN_verbose, info, primitive, exec, cpu, convolution, jitted:avx512, forward_inference, data:f32 blk, weights:f32 blk, 3x3, 1.25",
    "oneDNN_verbose, info, primitive, exec, cpu, inner_product, ref:any, forward_inference, data:f32, weights:f32, 0.75"
]

def test_parser_extracts_fields():
    r = parse_row(LOG_LINES[0])
    assert r["primitive"] == "convolution"
    assert r["time_ms"] == 1.25

def test_classifier_categorizes():
    assert classify_impl("jitted:avx512") == "x86_jit"
    assert classify_impl("ref:any") == "reference"

def test_reconcile_computes_correctly():
    rows = [parse_row(LOG_LINES[0]), parse_row(LOG_LINES[1])]
    res = reconcile_timing(rows, 3.0)
    assert res["total_kernel_time_ms"] == 2.0
    assert res["wall_clock_ms"] == 3.0
    assert res["discrepancy_ms"] == 1.0
