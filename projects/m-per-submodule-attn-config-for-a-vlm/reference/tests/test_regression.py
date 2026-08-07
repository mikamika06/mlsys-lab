import sys

sys.path.insert(0, ".")
from vlmcfg import build_configs, free_schedule, plan_bytes, uniform_bytes

CONFIG = {"submodules": [
    {"index": 0, "kind": "vision_proj", "num_heads": 16, "kv_heads": 16, "head_dim": 64, "causal": False},
    {"index": 1, "kind": "vision_proj", "num_heads": 16, "kv_heads": 16, "head_dim": 64, "causal": False},
    {"index": 2, "kind": "text_self", "num_heads": 32, "kv_heads": 8, "head_dim": 128, "causal": True},
    {"index": 3, "kind": "text_self", "num_heads": 32, "kv_heads": 8, "head_dim": 128, "causal": True}
]}
BY_INDEX = {s["index"]: s for s in CONFIG["submodules"]}


def test_no_group_mixes_two_configs():
    for g in build_configs(CONFIG):
        nh = {BY_INDEX[i]["num_heads"] for i in g["submodules"]}
        causal = {BY_INDEX[i]["causal"] for i in g["submodules"]}
        kinds = {BY_INDEX[i]["kind"] for i in g["submodules"]}
        assert len(nh) == 1, f"group {g['submodules']} spans num_heads {sorted(nh)}"
        assert len(causal) == 1, f"group {g['submodules']} spans causal {sorted(causal)}"
        assert len(kinds) == 1, f"group {g['submodules']} spans kinds {sorted(kinds)}"


def test_every_submodule_lands_in_exactly_one_group():
    seen = [i for g in build_configs(CONFIG) for i in g["submodules"]]
    assert sorted(seen) == sorted(BY_INDEX), f"{sorted(seen)} != {sorted(BY_INDEX)}"
    assert len(seen) == len(set(seen)), "a submodule appears in more than one group"


def test_grouping_never_costs_more_than_uniform():
    a = plan_bytes(CONFIG, 4096, 2, 1)
    b = uniform_bytes(CONFIG, 4096, 2, 1)
    assert a <= b, f"grouped plan wants {a} bytes, uniform only {b}"


def test_freed_blocks_never_go_backwards():
    s = free_schedule(4096, 2, 10)
    assert all(s[i] <= s[i + 1] for i in range(len(s) - 1)), "freed count decreased"
    assert s[0] == 0
