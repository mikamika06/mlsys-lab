import ref
import numpy as np


def check(workdir):
    from gptq.single import run_gptq_single

    W, invH, _ = ref.generate_inputs()
    want = ref.reference_gptq_single(W, invH, bits=4)
    got = run_gptq_single(W, invH, bits=4)
    out = {"max_abs_err": float("inf")}
    if got is None:
        out["_note"] = "run_gptq_single returned None"
        return out
    err = np.max(np.abs(got - want))
    out["max_abs_err"] = float(err)
    return out
