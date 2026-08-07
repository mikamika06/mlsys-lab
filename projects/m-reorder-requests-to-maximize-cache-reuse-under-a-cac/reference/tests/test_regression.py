import sys

sys.path.insert(0, ".")
from cacheopt.apc import apc_ttft_ratio
from cacheopt.analysis import identify_eviction
from cacheopt.reorder import reorder_requests


def test_apc_ratio_monotonicity():
    val_on = apc_ttft_ratio(4096, 4, True)
    val_off = apc_ttft_ratio(4096, 4, False)
    assert val_on < val_off


def test_reorder_preserves_length():
    reqs = [{"id": i, "prefix": [1, 2]} for i in range(5)]
    res = reorder_requests(reqs, 10)
    assert len(res) == len(reqs)


def test_identify_eviction_valid():
    reqs = [{"id": 1, "prefix": [1, 2]}, {"id": 2, "prefix": [1, 2, 3, 4, 5]}]
    res = identify_eviction(reqs, 3)
    assert res == 2
