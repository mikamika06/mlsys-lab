import sys
sys.path.insert(0, ".")
from radix.tree import TokenRadixTree
from radix.schedule import schedule_requests

def test_radix_insert_and_match():
    tree = TokenRadixTree()
    tree.insert([1, 2, 3, 4])
    matched, _ = tree.longest_match([1, 2, 3, 4, 5])
    assert matched == [1, 2, 3, 4]

def test_radix_split():
    tree = TokenRadixTree()
    tree.insert([1, 2, 3, 4])
    tree.insert([1, 2, 9, 10])
    matched, _ = tree.longest_match([1, 2, 9, 10])
    assert matched == [1, 2, 9, 10]

def test_schedule_lpm():
    reqs = [[1, 2, 3], [1, 2, 9], [1, 2, 3, 4]]
    res = schedule_requests(reqs, policy="lpm")
    assert len(res) == 3
