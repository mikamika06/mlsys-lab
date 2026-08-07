def check(workdir):
    import sys
    import os
    sys.path.insert(0, workdir)
    import ref
    import numpy as np
    from quant_rec.analysis import measure_layer_error

    m = {"layer_error_identified": 0.0}
    weights, quantized, _, _ = ref.get_test_data()
    err = measure_layer_error(weights, quantized)
    if isinstance(err, float) and err > 0.0:
        m["layer_error_identified"] = 1.0
    return m
