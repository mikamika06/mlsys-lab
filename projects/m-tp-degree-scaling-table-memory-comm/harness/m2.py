import ref
import numpy as np

def check(workdir):
    from tpscaling.rowparallel import row_parallel_forward
    out = {"max_rel_err": 0.0}
    np.random.seed(42)
    d_in, d_out = 512, 256
    weight = np.random.randn(d_in, d_out)
    bias = np.random.randn(d_out)
    x = np.random.randn(4, d_in)
    world_size = 4
    max_err = 0.0
    for rank in range(world_size):
        want = ref.row_parallel_forward(weight, bias, x, rank, world_size)
        got = row_parallel_forward(weight, bias, x, rank, world_size)
        err = np.max(np.abs(got - want) / (np.abs(want) + 1e-9))
        if err > max_err:
            max_err = err
    out["max_rel_err"] = float(max_err)
    return out
