import numpy as np
import ref


def check(workdir):
    from ringattn.ulysses import ulysses_attention

    q, k, v = ref.generate_ulysses_inputs()
    world_size = 4
    rank = 0
    num_heads = 8

    want = ref.ref_ulysses(q, k, v, rank, world_size, num_heads)
    try:
        got = ulysses_attention(q, k, v, rank, world_size, num_heads)
    except Exception as e:
        return {"ulysses_max_abs_err": float('inf'), "_note": f"raised {type(e).__name__}: {e}"}

    if got is None or np.shape(got) != np.shape(want):
        return {"ulysses_max_abs_err": float('inf'), "_note": f"shape mismatch got {np.shape(got) if got is not None else None}, want {np.shape(want)}"}

    err = float(np.max(np.abs(got - want)))
    return {"ulysses_max_abs_err": err}
