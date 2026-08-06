from pack.shapes import derive_shape_assertions


def test_shape_assertions_validity():
    cfg = {
        "model_id": "test_model",
        "hidden_dim": 64,
        "num_layers": 2,
        "supports_export": True,
        "cold_eager": 10.0,
        "cold_aot": 4.0,
        "warm_eager": 1.0,
        "warm_aot": 0.5,
        "min_seq": 1,
        "max_seq": 512,
    }
    res = derive_shape_assertions(cfg)
    assert isinstance(res, dict)
    assert res.get("valid") is True
    assert res.get("min_seq") <= res.get("max_seq")
    assert res.get("hidden_dim") > 0
