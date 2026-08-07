import ref
import numpy as np


def check(workdir):
    from mxquant.quant import quantize_linear

    w = ref.generate_test_data()
    group_size = 32
    bits = 4

    try:
        _, _, _, ratio = quantize_linear(w, group_size, bits)
    except Exception as e:
        return {"size_reduction_match": 0.0, "_note": f"Exception raised: {e}"}

    if isinstance(ratio, (int, float, np.number)) and ratio > 1.0:
        return {"size_reduction_match": 1.0}
    return {"size_reduction_match": 0.0, "_note": f"Invalid ratio returned: {ratio}"}
