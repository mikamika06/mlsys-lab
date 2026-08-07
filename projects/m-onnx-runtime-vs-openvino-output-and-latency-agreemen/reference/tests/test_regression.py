import sys
sys.path.insert(0, ".")

from bench.xeon_ranking import check_precision_fairness


def test_precision_fairness_rejects_mismatch():
    info_fp32 = {"precision": "fp32", "quantized": False}
    info_int8 = {"precision": "int8", "quantized": True}
    res = check_precision_fairness(info_fp32, info_int8)
    assert not res["fair"], "Fairness check must reject FP32 vs INT8 comparison"


def test_precision_fairness_accepts_matching():
    info_a = {"precision": "fp32", "quantized": False}
    info_b = {"precision": "fp32", "quantized": False}
    res = check_precision_fairness(info_a, info_b)
    assert res["fair"], "Fairness check must accept matching precisions"
