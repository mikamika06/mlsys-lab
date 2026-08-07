import sys

sys.path.insert(0, ".")
from kvradix.eviction import EvictableRadixCache
from kvradix.radix import RadixTree


def test_prefix_matching():
    tree = RadixTree()
    tree.insert([10, 20, 30, 40])
    matched, node, rem = tree.match_prefix([10, 20, 30, 50, 60])
    assert matched == 3
    assert rem == [50, 60]


def test_node_split():
    tree = RadixTree()
    tree.insert([1, 2, 3, 4, 5])
    tree.insert([1, 2, 3, 8, 9])
    matched, node, rem = tree.match_prefix([1, 2, 3, 8, 9])
    assert matched == 5
    assert len(tree.root.children) == 1
    root_child = next(iter(tree.root.children.values()))
    assert root_child.key == [1, 2, 3]


def test_eviction_under_capacity():
    cache = EvictableRadixCache(max_tokens=10)
    cache.insert_and_cache([1, 2, 3, 4, 5])
    cache.insert_and_cache([1, 2, 3, 6, 7])
    assert cache.total_tokens() <= 10
