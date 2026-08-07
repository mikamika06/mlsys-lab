import sys
import numpy as np

sys.path.insert(0, ".")
from checkpoint.merge import merge_tp_shards

def test_merge_respects_different_axes():
    shards = [
        {"row": np.zeros((2, 4)), "col": np.zeros((4, 2))},
        {"row": np.zeros((2, 4)), "col": np.zeros((4, 2))}
    ]
    axis_map = {"row": 1, "col": 0}
    out = merge_tp_shards(shards, axis_map)
    assert out["row"].shape == (2, 8)
    assert out["col"].shape == (8, 2)

def test_merge_handles_replication():
    shards = [
        {"rep": np.ones((2, 2)) * 1},
        {"rep": np.ones((2, 2)) * 2}
    ]
    axis_map = {"rep": None}
    out = merge_tp_shards(shards, axis_map)
    assert out["rep"].shape == (2, 2)
    assert np.all(out["rep"] == 1)
