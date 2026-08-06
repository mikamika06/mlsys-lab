import os
import tempfile
from marlin.eligibility import is_marlin_eligible
from marlin.validator import validate_quantize_config, QuantizationConfigError
from marlin.persistence import save_quantize_config, load_quantize_config

def test_marlin_eligibility_basic():
    cfg = {"bits": 4, "group_size": 128, "sym": True, "desc_act": False}
    assert is_marlin_eligible(cfg) is True

def test_validator_raises_on_bad_bits():
    cfg = {"bits": 5}
    try:
        validate_quantize_config(cfg)
        assert False, "Should have raised QuantizationConfigError"
    except QuantizationConfigError:
        pass

def test_round_trip_persistence():
    cfg = {"bits": 8, "group_size": 64, "sym": True, "desc_act": False, "quant_method": "marlin"}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "quantize_config.json")
        save_quantize_config(cfg, path)
        loaded = load_quantize_config(path)
        assert loaded["bits"] == cfg["bits"]
        assert loaded["group_size"] == cfg["group_size"]
        assert loaded["sym"] == cfg["sym"]
