import numpy as np
import ref


def check(workdir):
    from ringattn.ulysses import ulysses_reshuffle

    np.random.seed(42)
    x = np.random.randn(1, 4, 4, 8)
    world_size = 2

    want = ref.ulysses_reshuffle(x, world_size)
    got = ulysses_reshuffle(x, world_size)
    err = float(np.max(np.abs(want - got)))
    return {"max_abs_err": err}
