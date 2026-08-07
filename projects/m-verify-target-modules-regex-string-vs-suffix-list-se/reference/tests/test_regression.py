import sys

sys.path.insert(0, ".")
from loratarget.matcher import resolve_by_suffix
from loratarget.stats import verify_equivalence

TREE = {
    "model.layers.0.self_attn.proj": (4096, 4096),
    "model.layers.0.self_attn.q_proj": (4096, 4096),
    "model.layers.0.self_attn.v_proj": (4096, 4096),
}


def test_suffix_matching_exact_boundary():
    matched = resolve_by_suffix(TREE, ["proj"])
    assert "model.layers.0.self_attn.q_proj" not in matched
    assert matched == ["model.layers.0.self_attn.proj"]


def test_equivalence_detects_partial_substring_difference():
    is_equiv = verify_equivalence(TREE, ".*q_proj$", ["proj"])
    assert is_equiv is False
