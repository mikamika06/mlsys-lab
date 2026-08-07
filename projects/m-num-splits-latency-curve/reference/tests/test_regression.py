import sys
sys.path.insert(0, ".")

from splitkv.paged_plan import build_paged_split_plan


def test_paged_splits_cover_all_blocks():
    block_table = [10, 11, 12, 13, 14, 15, 16]
    page_size = 16
    splits = 3
    plan = build_paged_split_plan(block_table, page_size, splits)
    collected = []
    for p in plan:
        collected.extend(p["blocks"])
    assert collected == block_table, f"Expected {block_table}, got {collected}"


def test_paged_splits_strictly_monotonic_tokens():
    block_table = [10, 20, 30, 40, 50, 60]
    page_size = 32
    splits = 4
    plan = build_paged_split_plan(block_table, page_size, splits)
    curr = 0
    for p in plan:
        assert p["start_token"] == curr, f"Discontinuity at {p['start_token']} vs {curr}"
        assert p["end_token"] > p["start_token"], "Empty or inverted token range"
        curr = p["end_token"]
    assert curr == len(block_table) * page_size, f"Total tokens mismatch: {curr}"


def test_no_empty_splits_when_valid():
    block_table = [1, 2, 3]
    page_size = 64
    splits = 8
    plan = build_paged_split_plan(block_table, page_size, splits)
    assert len(plan) == len(block_table)
    for p in plan:
        assert len(p["blocks"]) > 0
