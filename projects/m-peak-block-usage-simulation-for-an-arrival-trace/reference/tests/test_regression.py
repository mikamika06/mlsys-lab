import math
import sys

sys.path.insert(0, ".")
from kvtrace.analysis import compute_paged_waste
from kvtrace.simulator import trace_block_timeline


def test_paged_blocks_never_exceed_ceiling():
    events = [
        {"req_id": 1, "arrival_time": 0, "prompt_len": 10, "gen_len": 5, "decode_speed": 1},
        {"req_id": 2, "arrival_time": 2, "prompt_len": 30, "gen_len": 10, "decode_speed": 1},
    ]
    block_size = 16
    timeline = trace_block_timeline(events, block_size)
    max_tokens = 10 + 5 + 30 + 10
    max_expected_blocks = math.ceil(max_tokens / block_size) + 2
    for t, blocks in timeline:
        assert blocks <= max_expected_blocks, f"Timeline blocks {blocks} exceeded bound {max_expected_blocks} at t={t}"


def test_timeline_non_negative_and_zero_at_end():
    events = [
        {"req_id": 1, "arrival_time": 0, "prompt_len": 10, "gen_len": 5, "decode_speed": 1},
    ]
    timeline = trace_block_timeline(events, 16)
    assert len(timeline) > 0
    for t, blocks in timeline:
        assert blocks >= 0, f"Negative block count {blocks} at time {t}"
    assert timeline[-1][1] == 0, f"Final block count is {timeline[-1][1]}, expected 0"


def test_paged_waste_non_negative():
    hist = {10: 5, 16: 2, 33: 10}
    waste = compute_paged_waste(hist, 16)
    assert waste >= 0, f"Paged waste was negative: {waste}"
