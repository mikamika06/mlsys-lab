import ref
import numpy as np

def check(workdir):
    from numerics.threshold import compute_clip_threshold
    grads = ref.synthetic_exploding_gradients()
    max_norm = 1.0
    want = ref.expected_threshold(grads, max_norm)
    got = compute_clip_threshold(grads, max_norm)
    if np.isclose(got, want, rtol=1e-5, atol=1e-5):
        return {"exact_match": 1.0}
    return {"exact_match": 0.0, "_note": f"got {got}, want {want}"}
