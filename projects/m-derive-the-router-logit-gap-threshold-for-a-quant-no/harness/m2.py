import ref
from routerquant.threshold import derive_logit_gap_threshold
import numpy as np


def check(workdir):
    _, weights, hidden_states, logits = ref.generate_inputs()
    quantized_weights = weights + 0.01
    want = derive_logit_gap_threshold(logits, weights, quantized_weights, hidden_states)
    from routerquant import threshold
    try:
        got = threshold.derive_logit_gap_threshold(logits, weights, quantized_weights, hidden_states)
    except Exception as e:
        return {"threshold_matched": 0.0, "_note": f"raised {type(e).__name__}"}
    if got is None or not hasattr(got, "shape"):
        return {"threshold_matched": 0.0, "_note": "invalid output"}
    if got.shape == want.shape and abs(np.sum(got - want)) < 1e-5:
        return {"threshold_matched": 1.0}
    return {"threshold_matched": 0.0, "_note": "threshold mismatch"}
