from lorascaling.predictor import predict_rank_requirements


def test_predictor_isolation():
    params = {
        "vram_base": 12e9,
        "vram_slope": 1.5e8,
        "flops_base": 8.0e14,
        "flops_slope": 2.5e12,
    }

    res1 = predict_rank_requirements(params, 8)
    res2 = predict_rank_requirements(params, 64)

    expected_vram1 = 12e9 + 1.5e8 * 8
    expected_vram2 = 12e9 + 1.5e8 * 64

    assert abs(res1["predicted_vram_bytes"] - expected_vram1) < 1.0
    assert abs(res2["predicted_vram_bytes"] - expected_vram2) < 1.0

    vram_ratio = res2["predicted_vram_bytes"] / res1["predicted_vram_bytes"]
    rank_ratio = 64.0 / 8.0

    assert abs(vram_ratio - rank_ratio) > 0.1
