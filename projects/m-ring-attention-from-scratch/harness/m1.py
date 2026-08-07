import numpy as np
import ref


def check(workdir):
    from ringattn.core import ring_attention

    q, k, v = ref.generate_inputs()
    world_size = 4
    rank = 0

    want = ref.ref_ring(q, k, v, rank, world_size)
    try:
        got = ring_attention(q, k, v, rank, world_size)
    except Exception as e:
        return {"ring_max_abs_err": float('inf'), "_note": f"raised {type(e).__name__}: {e}"}

    if got is None or np.shape(got) != np.shape(want):
        return {"ring_max_abs_err": float('inf'), "_note": f"shape mismatch got {np.shape(got) if got is not None else None}, want {np.shape(want)}"}

    err = float(np.max(np.abs(got - want)))
    return {"ring_max_abs_err": err}
