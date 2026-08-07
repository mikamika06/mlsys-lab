import numpy as np
import ref

def check(workdir):
    from ringattn.ring import ring_attention
    q, k, v, world_size = ref.generate_inputs()
    want = ref.reference_ring_attention(q, k, v, world_size)
    got = ring_attention(q, k, v, world_size)

    if got is None or got.shape != want.shape:
        return {"max_abs_err": float("inf"), "_note": f"shape mismatch: got {getattr(got, 'shape', None)}, want {want.shape}"}

    err = float(np.max(np.abs(want - got)))
    return {"max_abs_err": err}
