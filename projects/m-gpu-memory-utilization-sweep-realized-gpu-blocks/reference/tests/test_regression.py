import sys
sys.path.insert(0, ".")
from gpucache.multi import allocate_multi_model_memory

def test_multi_model_sum_exact():
    allocs = allocate_multi_model_memory(80 * 1024**3, 5 * 1024**3, [0.6, 0.4])
    net = 80 * 1024**3 - 5 * 1024**3
    assert sum(allocs) <= net

def test_multi_model_non_negative():
    allocs = allocate_multi_model_memory(40 * 1024**3, 2 * 1024**3, [0.5, 0.5])
    assert all(a >= 0 for a in allocs)
