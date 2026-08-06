import numpy as np
from engine.runner import execute_engine

def test_regression():
    engine = {"config": {"max_batch_size": 4, "fp16": False}}
    inputs = [np.ones((2, 4), dtype=np.float32)]
    res = execute_engine(engine, inputs)
    assert len(res) == 1
    assert np.allclose(res[0], 2.0)
