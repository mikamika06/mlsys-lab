import numpy as np
from mlxdist.pipeline import compute_pipeline_comm_volume
from mlxdist.ring import launch_2rank_ring_all_reduce
from mlxdist.sharding import derive_load_balanced_sharding


def test_pipeline_sharding_and_ring():
    """Test pipeline sharding bounds and ring reduction invariants."""
    assignments = [0, 0, 1, 1, 2, 3]
    shapes = [(1, 128, 4096)] * 5
    res = compute_pipeline_comm_volume(assignments, shapes, dtype_bytes=2)

    assert res["total_volume"] == 3 * (1 * 128 * 4096 * 2)
    assert sum(res["send_bytes"]) == sum(res["recv_bytes"])

    weights = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.0]
    sharding = derive_load_balanced_sharding(len(weights), weights, num_ranks=4)
    assert len(sharding) == len(weights)
    assert max(sharding) == 3
    assert min(sharding) == 0
    assert all(sharding[i] <= sharding[i + 1] for i in range(len(sharding) - 1))

    t0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    t1 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    ring_res = launch_2rank_ring_all_reduce(t0, t1)

    expected = t0 + t1
    np.testing.assert_allclose(ring_res["rank0"], expected)
    np.testing.assert_allclose(ring_res["rank1"], expected)
