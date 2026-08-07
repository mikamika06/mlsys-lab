import sys

sys.path.insert(0, ".")
from syncbug.sim import simulate_accumulation
from syncbug.bench import reduce_time_ratio
from syncbug.breakeven import compute_breakeven_k


def test_last_microbatch_syncs():
    res = simulate_accumulation(4, sync_last=True)
    assert res["synced"] is True, "Last microbatch failed to sync"


def test_reduce_ratio_positive():
    ratio = reduce_time_ratio(1000, 4, 2)
    assert ratio > 1.0


def test_breakeven_k_valid():
    k = compute_breakeven_k(2, 1e-8, 1e10)
    assert k >= 1
