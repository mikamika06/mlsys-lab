import ref
import numpy as np


def check(workdir):
    from bnb_quant.core import quantize_blockwise

    tensor = ref.generate_test_tensor()
    blocksizes = [32, 64, 128]
    matched = 0

    for bs in blocksizes:
        want = ref.reference_quantize(tensor, bs)
        try:
            got = quantize_blockwise(tensor, bs)
        except Exception:
            continue
        if got is None or not isinstance(got, dict):
            continue
        if "quantized" in got and "absmax" in got:
            q_match = np.array_equal(got["quantized"], want["quantized"])
            a_match = np.allclose(got["absmax"], want["absmax"], atol=1e-5)
            if q_match and a_match:
                matched += 1

    return {"blocksizes_matched": float(matched)}
