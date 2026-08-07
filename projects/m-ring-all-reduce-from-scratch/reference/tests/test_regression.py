import sys
sys.path.insert(0, ".")
from dist.bucket import GradientBucket
from dist.cost import ring_cost, tree_cost, find_crossover
from dist.ring import ring_all_reduce

def test_bucket_capacity_enforcement():
    bucket = GradientBucket(100)
    t1 = [1.0] * 10
    assert bucket.add(t1) is True
    t2 = [2.0] * 50
    assert bucket.add(t2) is False

def test_ring_correctness_simple():
    tensors = [[1.0, 2.0, 3.0, 4.0]]
    res = ring_all_reduce(tensors, rank=0, world_size=4)
    assert len(res) == 1
    assert len(res[0]) == 4

def test_crossover_ordering():
    alpha = 0.0001
    beta = 1e-9
    co = find_crossover(4, alpha, beta)
    assert co > 0
