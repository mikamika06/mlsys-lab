import sys
import numpy as np

sys.path.insert(0, ".")
from batchspmd.vmap import per_example_loop, verify_vmap_matches
from batchspmd.benchmark import benchmark_vmap_speedup
from batchspmd.spmd import simulated_psum, simulated_pmap, spmd_allreduce_step


def test_vmap_matches_loop_exact():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    fn_s = lambda v: v * 2.0 + 1.0
    fn_b = lambda m: m * 2.0 + 1.0
    err = verify_vmap_matches(fn_s, fn_b, x)
    assert err < 1e-5, f"Expected error < 1e-5, got {err}"


def test_spmd_allreduce_all_devices_identical():
    x = np.ones((8, 4), dtype=np.float32)
    x[4:] = 3.0
    fn = lambda shard: np.sum(shard, axis=0)
    res = spmd_allreduce_step(x, fn, num_devices=4)
    for i in range(1, res.shape[0]):
        assert np.allclose(res[0], res[i]), f"Device 0 and Device {i} outputs differ in all-reduce"


def test_spmd_allreduce_sum_value():
    x = np.ones((4, 2), dtype=np.float32)
    fn = lambda shard: np.mean(shard, axis=0)
    res = spmd_allreduce_step(x, fn, num_devices=4)
    assert np.allclose(res[0], [4.0, 4.0]), f"Expected [4.0, 4.0], got {res[0]}"
