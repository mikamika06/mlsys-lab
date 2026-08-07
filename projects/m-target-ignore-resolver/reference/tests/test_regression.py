import sys

sys.path.insert(0, ".")
from quantres.resolver import resolve_targets
from quantres.moe import build_moe_ignore_list
from quantres.router import find_wrong_router


def test_resolver_basic():
    mods = ["layer.0.attn.q", "layer.0.mlp.gate"]
    targets = ["*"]
    ignores = ["*q"]
    res = resolve_targets(mods, targets, ignores)
    assert res == ["layer.0.mlp.gate"]


def test_moe_ignore_contains_gate():
    struct = {"modules": ["model.router.weight", "model.linear.weight"]}
    ignores = build_moe_ignore_list(struct)
    assert "model.router.weight" in ignores


def test_wrong_router_detection():
    q_mods = {"model.router": "int4", "model.linear": "int8"}
    wrong = find_wrong_router(q_mods)
    assert "model.router" in wrong
