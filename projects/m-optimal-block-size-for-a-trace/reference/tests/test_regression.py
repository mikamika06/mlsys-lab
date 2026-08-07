"""Regression tests for kvblock."""

import sys

sys.path.insert(0, ".")
from kvblock.cache import PrefixCacheSimulator
from kvblock.trace import find_optimal_block_size, total_overhead
from kvblock.triage import triage_block_table


def test_optimal_block_size():
    trace = [128, 256, 512, 1024]
    candidates = [8, 16, 32, 64, 128]
    costs = total_overhead(trace, candidates)
    best = find_optimal_block_size(trace, candidates)
    assert best in candidates
    assert costs[candidates.index(best)] == min(costs)


def test_prefix_cache_simulation():
    sim = PrefixCacheSimulator(block_size=4, max_blocks=10)
    req1 = [1, 2, 3, 4, 5, 6, 7, 8]
    req2 = [1, 2, 3, 4, 9, 10, 11, 12]
    hits1, num1 = sim.process_request(req1)
    hits2, num2 = sim.process_request(req2)
    assert hits1 == 0
    assert hits2 == 1
    assert sim.hit_rate() == 1.0 / 4.0


def test_triage_corrupt_block_table():
    bt = [0, 1, 999, 1]
    res = triage_block_table(bt, total_seq_len=64, block_size=16, max_valid_block_id=100)
    assert res["is_valid"] is False
    assert res["expected_blocks"] == 4
    assert res["repaired_table"][2] == -1
    assert res["repaired_table"][3] == -1
