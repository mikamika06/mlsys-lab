import sys
import numpy as np

sys.path.insert(0, ".")
from moe_routing.comm import all_to_all_shapes


def test_all_to_all_symmetry():
    assignments = np.array([[0, 2], [1, 5], [2, 0], [3, 7]])
    send_counts, recv_counts = all_to_all_shapes(assignments, 8, 4)
    assert np.array_equal(send_counts.T, recv_counts), "Receive shapes must be exactly the transpose of send shapes"
    assert np.sum(send_counts) == 4, "Total send counts must equal number of routed tokens"


def test_all_to_all_empty():
    assignments = np.array([]).reshape(0, 2).astype(int)
    send_counts, recv_counts = all_to_all_shapes(assignments, 8, 4)
    assert np.array_equal(send_counts.T, recv_counts)
    assert np.sum(send_counts) == 0
