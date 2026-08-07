import sys

sys.path.insert(0, ".")
from fp8read.analysis import compute_ratios
from fp8read.parser import parse_logs


def test_parse_logs_basic():
    logs = ["SHAPE M=128 N=128 K=128 FP8_TFLOPS=50.0 FP16_CUBLAS_TFLOPS=40.0"]
    res = parse_logs(logs)
    assert len(res) == 1
    assert res[0]["M"] == 128


def test_compute_ratios_positive():
    records = [{"M": 128, "N": 128, "K": 128, "FP8_TFLOPS": 50.0, "FP16_CUBLAS_TFLOPS": 25.0}]
    res = compute_ratios(records)
    assert res[0]["throughput_ratio"] == 2.0
