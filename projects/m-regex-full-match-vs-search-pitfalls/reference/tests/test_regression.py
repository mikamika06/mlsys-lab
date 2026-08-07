import sys

sys.path.insert(0, ".")
from modfilter.matcher import is_matched, filter_modules
from modfilter.rules import apply_rules


def test_no_substring_false_positives():
    assert not is_matched("mlp", "mlp_proj")
    assert not is_matched("attn", "self_attn")
    assert is_matched("mlp", "mlp")


def test_filter_modules_strict():
    mods = ["mlp", "mlp_proj", "self_attn", "attn"]
    res = filter_modules(mods, ["mlp", "attn"], [])
    assert res == ["mlp", "attn"]


def test_rules_application():
    mods = ["layer.0.mlp", "layer.0.mlp_proj", "layer.1.attn"]
    rules = [{"action": "include", "patterns": [r"layer\.\d+\.mlp"]}]
    res = apply_rules(mods, rules)
    assert res == ["layer.0.mlp"]
