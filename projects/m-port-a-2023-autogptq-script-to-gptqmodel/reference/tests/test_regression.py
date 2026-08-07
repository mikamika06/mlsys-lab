import sys
sys.path.insert(0, ".")
from gptq_port.translate import translate_config
from gptq_port.oracle import check_compatibility
from gptq_port.classify import identify_library

def test_translate_preserves_core_fields():
    legacy = {"bits": 4, "group_size": 128, "desc_act": True, "sym": False}
    out = translate_config(legacy)
    assert out["bits"] == 4
    assert out["group_size"] == 128
    assert out["desc_act"] is True
    assert out["sym"] is False
    assert out["quant_method"] == "gptq"

def test_oracle_rejects_invalid_exllamav2():
    cfg = {"bits": 5, "group_size": 64, "desc_act": False}
    assert check_compatibility(cfg, "exllamav2") is False

def test_classify_identifies_gptqmodel():
    files = ["config.json", "quantize_config.json", "model.safetensors", "gptqmodel_metadata.json"]
    assert identify_library(files) == "gptqmodel"
