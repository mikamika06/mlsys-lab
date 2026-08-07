import ref
import numpy as np


def check(workdir):
    from jax_shape_lab.shard_compare import simulate_shard_vs_pmap

    arr = np.ones((4, 4), dtype=np.float32)
    got = simulate_shard_vs_pmap(arr, "x")
    want = ref.simulate_shard_vs_pmap(arr, "x")
    if np.allclose(got, want):
        return {"parity_matched": 1.0}
    return {"parity_matched": 0.0}
