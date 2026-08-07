import sys
sys.path.insert(0, ".")
from profiler.stats import aggregate_cupti

def test_cupti_aggregation_structure():
    rows = [
        {"name": "gemm_kernel", "start": 100, "end": 200},
        {"name": "gemm_kernel", "start": 300, "end": 450},
        {"name": "relu_kernel", "start": 500, "end": 550}
    ]
    res = aggregate_cupti(rows)
    assert "gemm_kernel" in res
    assert res["gemm_kernel"]["count"] == 2
    assert res["gemm_kernel"]["total"] == 250.0
    assert res["gemm_kernel"]["avg"] == 125.0
    assert res["gemm_kernel"]["min"] == 100.0
    assert res["gemm_kernel"]["max"] == 150.0
