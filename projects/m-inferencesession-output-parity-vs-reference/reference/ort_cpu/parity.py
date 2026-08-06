import numpy as np
import onnxruntime as ort


def verify_parity(model_bytes, inputs, ref_outputs, rel_err_tol=1e-5):
    opts = ort.SessionOptions()
    session = ort.InferenceSession(model_bytes, opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    for x, ref_out in zip(inputs, ref_outputs):
        outputs = session.run(None, {input_name: x})
        got = outputs[0]
        rel_err = np.max(np.abs(got - ref_out) / (np.abs(ref_out) + 1e-8))
        if rel_err > rel_err_tol:
            return False
    return True
