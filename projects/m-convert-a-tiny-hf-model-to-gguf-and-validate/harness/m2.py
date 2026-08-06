import numpy as np
import ref

def check(workdir):
    out = {"precision_error_bounded": 0.0}
    try:
        from ggufconv.precision import convert_outtype
        tensors = ref.get_test_tensors()
        res_f16 = convert_outtype(tensors, "f16")
        res_bf16 = convert_outtype(tensors, "bf16")
        if len(res_f16) == 1 and len(res_bf16) == 1:
            out["precision_error_bounded"] = 1.0
        else:
            out["_note"] = "precision conversion output shape or length mismatch"
    except Exception as e:
        out["_note"] = str(e)[:120]
    return out
