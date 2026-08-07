import sys
sys.path.insert(0, ".")
from zeromem.buckets import compute_bucket_count
from zeromem.formula import compute_memory_table
from zeromem.reducescatter import toy_reduce_scatter

def test_bucket_count_basic():
    assert compute_bucket_count(1000, 500) == 2
    assert compute_bucket_count(1000, 1000) == 1
    assert compute_bucket_count(1000, 200) == 5

def test_formula_scaling():
    t1 = compute_memory_table(1000000, 4, 4)
    t2 = compute_memory_table(1000000, 8, 4)
    assert t2["optimizer_states"] < t1["optimizer_states"]

def test_reduce_scatter_correctness():
    grads = [10.0, 20.0, 30.0, 40.0]
    res0 = toy_reduce_scatter(grads, 2, 0)
    res1 = toy_reduce_scatter(grads, 2, 1)
    assert res0 == [5.0, 10.0]
    assert res1 == [15.0, 20.0]
