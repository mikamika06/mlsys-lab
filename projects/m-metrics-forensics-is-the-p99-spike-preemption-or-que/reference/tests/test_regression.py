import sys

sys.path.insert(0, ".")
from forensics.capacity import calculate_zero_preemption_max_seqs


def test_zero_preemption_respects_block_limit():
    profile = {"qps": 10.0, "avg_tokens": 128, "block_size": 16}
    total_blocks = 100
    res = calculate_zero_preemption_max_seqs(profile, total_blocks)
    blocks_per_req = (128 + 15) // 16
    max_allowed = total_blocks // blocks_per_req
    assert res <= max_allowed, f"Returned max_seqs {res} exceeds physical block limit {max_allowed}"


def test_zero_preemption_positive():
    profile = {"qps": 5.0, "avg_tokens": 64, "block_size": 16}
    total_blocks = 200
    res = calculate_zero_preemption_max_seqs(profile, total_blocks)
    assert res > 0, f"Expected positive max_seqs, got {res}"
