import numpy as np
import ref


def check(workdir):
    from ortinfer.session import run_inference
    model_bytes = ref.get_model_bytes()
    inputs = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    got = run_inference(model_bytes, inputs, intra_op_num_threads=1, opt_level="BASIC")
    want = ref.reference_inference(inputs)
    err = float(np.max(np.abs(got - want) / (np.abs(want) + 1e-7)))
    out = {"rel_err": err}
    if err > 1e-5:
        out["_note"] = f"output mismatch: max relative error {err}"
    return out
