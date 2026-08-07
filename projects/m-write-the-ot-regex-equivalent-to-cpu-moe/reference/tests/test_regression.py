import sys
sys.path.insert(0, ".")
from ot.translator import translate_cpu_moe
from ot.validator import validate_regex

def test_translation_not_trivial():
    res = translate_cpu_moe("--cpu-moe")
    assert res != ".*", "Translation returned overly broad wildcard"
    assert "ffn_" in res, "Translation missing MoE component"

def test_validator_rejects_broad_patterns():
    assert not validate_regex(".*")
    assert not validate_regex("^blk\\..*")

def test_validator_accepts_valid_moe_regex():
    valid = "^blk\\.\\d+\\.ffn_(gate|up|down)_expts\\..*"
    assert validate_regex(valid)
