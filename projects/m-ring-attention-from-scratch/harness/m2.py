import numpy as np
import ref


def check(workdir):
    from ringattn.ulysses import ulysses_all_to_all

    rng = np.random.default_rng(42)
    b, s, h, d = 2, 16, 4, 8
    world_size = 4

    x = rng.standard_normal((b, s, h, d)).astype(np.float32)

    ref_out = ref.reference_ulysses_all_to_all(x, world_size)
    got_out = ulysses_all_to_all(x, world_size)

    max_err = float(np.max(np.abs(ref_out - got_out)))
    return {"max_abs_err": max_err}
