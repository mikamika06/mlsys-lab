import numpy as np
from quantizer.core import quantize_and_dequantize


def granularity_ladder_analysis(tensor, symmetric=True, num_bits=8, block_size=32, subvec_size=16):
    arr = np.asarray(tensor, dtype=np.float32)
    num_elements = arr.size

    grans = [
        ("per_tensor", None),
        ("per_channel", None),
        ("block_wise", block_size),
        ("sub_vector", subvec_size),
    ]

    results = []
    for g_name, g_size in grans:
        deq = quantize_and_dequantize(arr, g_name, symmetric=symmetric, num_bits=num_bits, group_size=g_size)
        err = float(np.max(np.abs(arr - deq)))

        if g_name == "per_tensor":
            num_params = 1
        elif g_name == "per_channel":
            num_params = arr.shape[0]
        elif g_name == "block_wise":
            num_params = num_elements // block_size
        elif g_name == "sub_vector":
            num_params = num_elements // subvec_size

        bytes_per_param = 4 if symmetric else 5
        metadata_bytes = num_params * bytes_per_param

        results.append({
            "granularity": g_name,
            "max_abs_err": err,
            "metadata_bytes": metadata_bytes,
        })

    return results
