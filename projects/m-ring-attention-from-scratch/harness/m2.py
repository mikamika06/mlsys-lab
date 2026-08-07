import numpy as np
import ref

def check(workdir):
    from ringattn.ulysses import ulysses_reshuffle
    q, k, v, world_size = ref.generate_inputs()
    b, s, h, d = q.shape
    s_p = s // world_size
    h_p = h // world_size

    rank = 0
    x_local = q[:, rank * s_p : (rank + 1) * s_p, :, :]

    want = ref.reference_ulysses_reshuffle(x_local, world_size, rank, forward=True)
    got = ulysses_reshuffle(x_local, world_size, rank, forward=True)

    if got is None or got.shape != want.shape:
        return {"max_abs_err": float("inf"), "_note": f"shape mismatch: got {getattr(got, 'shape', None)}, want {want.shape}"}

    err = float(np.max(np.abs(want - got)))
    return {"max_abs_err": err}
