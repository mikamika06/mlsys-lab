import sys
sys.path.insert(0, ".")
import numpy as np
from gguf_shard.model import Model
from gguf_shard.sharder import split, reassemble

def test_missing_shard_raises():
    m = Model({"arch": "test"}, {"w1": np.ones((10,)), "w2": np.ones((10,))})
    shards = split(m, 100)
    try:
        reassemble(shards[:1])
        assert False, "Should raise ValueError on missing shard"
    except ValueError:
        pass
