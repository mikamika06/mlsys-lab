import numpy as np
import ref


def check(workdir):
    from ringattn.devices import ring_attention
    q, k, v = ref.get_test_cases()
    got = ring_attention(q, k, v, num_devices=4)
    want = ref.naive_full_attention(q, k, v)
    err = float(np.max(np.abs(got - want)))
    return {"max_abs_err": err}
