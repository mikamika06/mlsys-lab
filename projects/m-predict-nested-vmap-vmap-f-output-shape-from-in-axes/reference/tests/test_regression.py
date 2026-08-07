import sys
sys.path.insert(0, ".")
from jax_shape_lab.predict import predict_nested_vmap_shape
from jax_shape_lab.shard_compare import simulate_shard_vs_pmap
import ref
import numpy as np


def test_predict_nested_vmap_shape_valid():
    for tc in ref.TEST_CASES:
        got = predict_nested_vmap_shape(
            tc["input_shape"],
            tc["in_axes_outer"],
            tc["in_axes_inner"],
            tc["base_out_shape"],
            tc["batch_outer"],
            tc["batch_inner"],
        )
        want = ref.predict_nested_vmap_shape(
            tc["input_shape"],
            tc["in_axes_outer"],
            tc["in_axes_inner"],
            tc["base_out_shape"],
            tc["batch_outer"],
            tc["batch_inner"],
        )
        assert got == want


def test_simulate_shard_vs_pmap_valid():
    arr = np.ones((4, 4), dtype=np.float32)
    got = simulate_shard_vs_pmap(arr, "x")
    want = np.sum(arr, axis=0, keepdims=True)
    assert np.allclose(got, want)
