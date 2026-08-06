from loraparams.compare import audit_recorded_run
from loraparams.formula import calculate_trainable_params


def test_lora_bias_calculation():
    model_cfg = {
        "modules": {
            "q_proj": {"in_dim": 128, "out_dim": 128, "count": 2, "has_bias": True},
            "k_proj": {"in_dim": 128, "out_dim": 128, "count": 2, "has_bias": True},
        }
    }
    lora_cfg = {
        "r": 4,
        "target_modules": ["q_proj"],
        "bias": "lora_only",
    }
    res = calculate_trainable_params(model_cfg, lora_cfg)
    assert res["lora_adapter_params"] == 2 * 4 * (128 + 128)
    assert res["bias_params"] == 2 * 128
    assert res["total_trainable_params"] == 2048 + 256


def test_modules_to_save_calculation():
    model_cfg = {
        "modules": {
            "q_proj": {"in_dim": 64, "out_dim": 64, "count": 1, "has_bias": False},
            "embed_tokens": {"in_dim": 100, "out_dim": 64, "count": 1, "has_bias": False},
        }
    }
    lora_cfg = {
        "r": 2,
        "target_modules": ["q_proj"],
        "modules_to_save": ["embed_tokens"],
    }
    res = calculate_trainable_params(model_cfg, lora_cfg)
    assert res["modules_to_save_params"] == 100 * 64
    assert res["total_trainable_params"] == 1 * 2 * (64 + 64) + 6400


def test_audit_recorded_run_mismatch():
    model_cfg = {
        "modules": {
            "v_proj": {"in_dim": 32, "out_dim": 32, "count": 1, "has_bias": False},
        }
    }
    lora_cfg = {"r": 4, "target_modules": ["v_proj"]}
    rec_matching = {
        "run_id": "run-1",
        "model_config": model_cfg,
        "lora_config": lora_cfg,
        "reported_trainable_params": 256,
    }
    rec_bad = {
        "run_id": "run-2",
        "model_config": model_cfg,
        "lora_config": lora_cfg,
        "reported_trainable_params": 512,
    }
    a1 = audit_recorded_run(rec_matching)
    a2 = audit_recorded_run(rec_bad)
    assert a1["is_valid"] is True
    assert a2["is_valid"] is False
    assert a2["delta"] == 256
