import sys

sys.path.insert(0, ".")
from trtmatrix.vc_cost import analyze_vc_cost_tradeoff


def test_vc_overhead_included_in_cost_analysis():
    models = [
        {
            "name": "bert_test",
            "base_bytes": 1000,
            "tactics_bytes": 100,
            "precision_scale": 1.0,
            "enable_refit": True,
        }
    ]
    res = analyze_vc_cost_tradeoff(
        models,
        trt_version_count=5,
        vc_overhead_bytes=500,
        refit_overhead_bytes=200,
    )
    m1_res = res["per_model"]["bert_test"]
    assert m1_res["std_engine_bytes"] == 1100
    assert (
        m1_res["vc_engine_bytes"] == 1800
    ), f"Expected 1800 bytes, got {m1_res['vc_engine_bytes']}"
    assert m1_res["total_std_storage_bytes"] == 5500
    assert m1_res["total_vc_storage_bytes"] == 1800
    assert m1_res["net_bytes_saved"] == 3700
