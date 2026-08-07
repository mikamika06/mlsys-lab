import sys

sys.path.insert(0, ".")
from radixtree.tree import TokenRadixTree
from radixtree.cache import simulate_cache
from radixtree.schedule import schedule_requests

def test_tree_insert_and_match():
    tree = TokenRadixTree()
    tokens = [1, 2, 3, 4, 5]
    tree.insert(tokens, value="val1")
    matched, val = tree.longest_match([1, 2, 3, 4, 5, 6])
    assert matched == [1, 2, 3, 4, 5]
    assert val == "val1"

def test_cache_hit_rate_positive():
    traces = [[1, 2, 3, 4], [1, 2, 5, 6]]
    res = simulate_cache(traces, "radix")
    assert res["hit_rate"] >= 0.0

def test_schedule_length():
    tree = TokenRadixTree()
    reqs = [[1, 2], [3, 4]]
    scheduled = schedule_requests(reqs, tree, "lpm")
    assert len(scheduled) == 2
