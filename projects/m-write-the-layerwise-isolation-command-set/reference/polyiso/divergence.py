import numpy as np


def find_first_divergent_layer(layer_outputs, rtol=1e-3, atol=1e-5):
    for layer_name, (out_ref, out_candidate) in layer_outputs.items():
        ref_arr = np.asarray(out_ref, dtype=np.float64)
        cand_arr = np.asarray(out_candidate, dtype=np.float64)
        if not np.allclose(ref_arr, cand_arr, rtol=rtol, atol=atol):
            return layer_name
    return None
