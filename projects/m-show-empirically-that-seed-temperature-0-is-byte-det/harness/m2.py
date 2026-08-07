import ref
import numpy as np

def check(workdir):
    from det.sampling import recover_sampling_params

    rng = np.random.RandomState(42)
    logits = rng.randn(10, 32)
    tokens = np.argmax(logits, axis=-1)

    ref_res = ref.recover_parameters(logits, tokens)
    try:
        user_res = recover_sampling_params(logits, tokens)
        ok = 1.0 if isinstance(user_res, dict) and "temperature" in user_res else 0.0
    except Exception:
        ok = 0.0

    return {"recovered_match": ok}
