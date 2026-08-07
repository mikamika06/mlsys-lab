import ref
import numpy as np


def check(workdir):
    from dist.ring import ring_all_reduce

    tensors = [[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0]]
    world_size = 4
    max_err = 0.0

    for rank in range(world_size):
        got = ring_all_reduce(tensors, rank, world_size)
        want = ref.reference_ring_all_reduce(tensors, rank, world_size)

        for g, w in zip(got, want):
            err = np.max(np.abs(np.array(g) - np.array(w)))
            if err > max_err:
                max_err = err

    out = {"max_abs_err": float(max_err)}
    return out
