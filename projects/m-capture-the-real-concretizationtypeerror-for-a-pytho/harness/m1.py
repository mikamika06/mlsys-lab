import numpy as np
import ref

def check(workdir):
    from jaxcomp.tracer import Tracer, capture_concretization_error, ConcretizationTypeError

    out = {"captured_correctly": 0.0}

    def bad_fn(x):
        if x:
            return 1
        return 0

    t = Tracer("x", (2, 2), np.float32)
    caught, err = capture_concretization_error(bad_fn, t)

    ref_caught, ref_err = ref.capture_concretization_error(bad_fn, t)

    if caught and isinstance(err, ConcretizationTypeError) and err.tracer is t:
        out["captured_correctly"] = 1.0
    else:
        out["_note"] = f"Expected ConcretizationTypeError with tracer attribute, got caught={caught}, err={err}"

    return out
