import sys

sys.path.insert(0, ".")
from kvplan import build_groups, free_schedule, plan_bytes, uniform_bytes

CONFIG = {"layers": [
    {"index": 0, "kind": "full", "kv_heads": 8, "head_dim": 128},
    {"index": 1, "kind": "sliding", "window": 512, "kv_heads": 8, "head_dim": 128},
    {"index": 2, "kind": "sliding", "window": 512, "kv_heads": 8, "head_dim": 128},
    {"index": 3, "kind": "sliding", "window": 2048, "kv_heads": 8, "head_dim": 128},
    {"index": 4, "kind": "full", "kv_heads": 8, "head_dim": 128},
]}
BY_INDEX = {l["index"]: l for l in CONFIG["layers"]}


def test_no_group_mixes_two_windows():
    for g in build_groups(CONFIG):
        windows = {BY_INDEX[i].get("window") or 0 for i in g["layers"]}
        kinds = {BY_INDEX[i]["kind"] for i in g["layers"]}
        assert len(windows) == 1, f"group {g['layers']} spans windows {sorted(windows)}"
        assert len(kinds) == 1, f"group {g['layers']} spans kinds {sorted(kinds)}"
        assert (g.get("window") or 0) in windows


def test_every_layer_lands_in_exactly_one_group():
    seen = [i for g in build_groups(CONFIG) for i in g["layers"]]
    assert sorted(seen) == sorted(BY_INDEX), f"{sorted(seen)} != {sorted(BY_INDEX)}"
    assert len(seen) == len(set(seen)), "a layer appears in more than one group"


def test_grouping_never_costs_more_than_uniform():
    a = plan_bytes(CONFIG, 8192, 16, 2)
    b = uniform_bytes(CONFIG, 8192, 16, 2)
    assert a <= b, f"grouped plan wants {a} bytes, uniform only {b}"


def test_freed_blocks_never_go_backwards():
    s = free_schedule(512, 16, 2000)
    assert all(s[i] <= s[i + 1] for i in range(len(s) - 1)), "freed count decreased"
    assert s[0] == 0
