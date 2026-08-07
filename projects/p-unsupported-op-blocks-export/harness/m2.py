import sys
import numpy as np
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    from exporter.replacements import substitute_op
    m = {"equivalent_found": 0.0}
    data = ref.get_test_data()
    out_ref = ref.reference_gelu(data)
    out_sub = substitute_op(data)
    if np.allclose(out_ref, out_sub, atol=1e-5):
        m["equivalent_found"] = 1.0
    return m
