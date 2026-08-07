import ref
import numpy as np


def check(workdir):
    from ggufparser.quant import tensor_byte_size

    types_to_test = [ref.GGML_TYPE_F16, ref.GGML_TYPE_Q8_0, ref.GGML_TYPE_Q4_K]
    ok = True

    for q in types_to_test:
        for dims in [[256, 128], [512], [1024, 64]]:
            n_el = int(np.prod(dims))
            want = ref.compute_tensor_bytes(n_el, q)
            got = tensor_byte_size(dims, q)
            if got != want:
                ok = False
                break
        if not ok:
            break

    return {"sizes_matched": 1.0 if ok else 0.0}
