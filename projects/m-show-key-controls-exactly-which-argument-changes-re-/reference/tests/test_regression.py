import sys
sys.path.insert(0, ".")
from triton_tune.autokey import check_key_triggers, find_true_argmin, measure_search_overhead


def test_key_triggers_basic():
    key_args = ["M", "N"]
    seqs = [
        {"M": 128, "N": 128, "K": 64},
        {"M": 128, "N": 128, "K": 128},
        {"M": 256, "N": 128, "K": 128},
    ]
    res = check_key_triggers(key_args, seqs)
    assert res == [True, False, True]


def test_find_true_argmin():
    records = [
        {"config": {"BLOCK_SIZE": 32}, "latency": 15.5},
        {"config": {"BLOCK_SIZE": 64}, "latency": 10.2},
        {"config": {"BLOCK_SIZE": 128}, "latency": 12.1},
    ]
    best = find_true_argmin(records)
    assert best["BLOCK_SIZE"] == 64


def test_measure_search_overhead():
    times = [1.0, 2.0, 3.0]
    hardcoded = 2.0
    ratio = measure_search_overhead(times, hardcoded)
    assert ratio == 3.0
