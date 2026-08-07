def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    import numpy as np
    from quant_rec.analysis import measure_layer_error

    m = {"accuracy_threshold_met": 0.0}
    weights, quantized, _, _ = ref.get_test_data()
    err = measure_layer_error(weights, quantized)
    if err < 5.0:
        m["accuracy_threshold_met"] = 1.0
    return m
