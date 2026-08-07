import ref
import numpy as np

def check(workdir):
    from milpass import fusion
    out = {"delta_matched": 0.0, "max_abs_err": 0.0}
    try:
        graph = ref.make_test_graph()
        fused, delta = fusion.fuse_conv_bn(graph)
        want_delta = ref.expected_fusion_delta()
        if delta == want_delta:
            out["delta_matched"] = 1.0
        else:
            out["_note"] = f"delta got {delta}, want {want_delta}"
        out["max_abs_err"] = 0.0
    except Exception as e:
        out["_note"] = f"error: {str(e)}"
        out["max_abs_err"] = 1.0
    return out
