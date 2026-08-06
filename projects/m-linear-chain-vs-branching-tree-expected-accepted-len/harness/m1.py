import ref
import numpy as np

def check(workdir):
    from speculative.metrics import expected_accepted_length_linear

    out = {"linear_rel_err": 1.0}
    errors = []

    for case in ref.TEST_LINEAR_CASES:
        probs = case["probs"]
        want = ref.expected_accepted_length_linear(probs)
        try:
            got = expected_accepted_length_linear(probs)
            err = abs(got - want) / max(1e-6, abs(want))
            errors.append(err)
        except Exception as e:
            out["_note"] = f"Error evaluating linear chain: {type(e).__name__}: {e}"
            return out

    if errors:
        out["linear_rel_err"] = float(np.mean(errors))
    return out
