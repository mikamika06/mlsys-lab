import numpy as np
import sys

sys.path.insert(0, ".")
from launchgraph.harness import StaticBufferHarness


def test_static_buffer_invariance():
    max_shape = (10, 10)
    harness = StaticBufferHarness(max_shape, dtype=np.float32)
    addr_history = []

    def dummy_graph_runner(static_buf):
        addr_history.append(static_buf.__array_interface__['data'][0])
        return static_buf * 2.0

    in1 = np.ones((5, 5), dtype=np.float32)
    harness.update_input(in1)
    res1 = harness.run(dummy_graph_runner)
    assert res1.shape == (5, 5)
    assert np.allclose(res1, 2.0)

    in2 = np.ones((8, 3), dtype=np.float32) * 3.0
    harness.update_input(in2)
    res2 = harness.run(dummy_graph_runner)
    assert res2.shape == (8, 3)
    assert np.allclose(res2, 6.0)

    assert len(addr_history) == 2
    assert addr_history[0] == addr_history[1], "Static buffer address changed during dynamic updates!"
