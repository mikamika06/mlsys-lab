import sys
sys.path.insert(0, ".")
from streamkv.selection import streaming_llm_indices
from streamkv.h2o import h2o_heavy_hitters
from streamkv.snapkv import snapkv_pool_scores

def test_streaming_llm_bounds():
    res = streaming_llm_indices(100, 4, 10)
    assert len(res) == 14
    assert res[:4] == [0, 1, 2, 3]
    assert res[-10:] == list(range(90, 100))

def test_h2o_capacity():
    matrix = [[0.1, 0.5, 0.2], [0.2, 0.4, 0.4]]
    res = h2o_heavy_hitters(matrix, 2)
    assert len(res) == 2
    assert sorted(res) == [1, 2]

def test_snapkv_selection():
    matrix = [[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 10.0]]
    res = snapkv_pool_scores(matrix, 4, 2)
    assert len(res) == 2
    assert res == [2, 3]
