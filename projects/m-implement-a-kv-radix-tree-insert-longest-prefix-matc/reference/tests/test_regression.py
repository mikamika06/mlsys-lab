from kvtree.tree import RadixTree
from kvtree.eviction import RadixEvictor

def test_radix_tree_insertion_and_match():
    tree = RadixTree()
    tree.insert([1, 2, 3], [10, 11])
    blocks, length = tree.match_prefix([1, 2, 3, 4])
    assert length == 3
    assert blocks == [10, 11]

def test_eviction_under_capacity():
    tree = RadixTree()
    evictor = RadixEvictor(tree, max_blocks=1)
    evictor.on_insert([10])
    evictor.on_insert([11])
    assert 10 in evictor.evicted_blocks
