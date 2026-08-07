import numpy as np
import ref


def check(workdir):
    from transformer.split_softmax import compute_split_softmax

    out = {"softmax_match": 0.0}
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2, 8, 32, 64))

    try:
        ref_out = ref.standard_softmax(x)
        user_parts = compute_split_softmax(x, chunks=4)
        if not isinstance(user_parts, list):
            out["_note"] = "compute_split_softmax must return a list of tensor chunks"
            return out
        user_out = np.concatenate(user_parts, axis=-1)

        if not np.allclose(user_out, ref_out, atol=1e-5, rtol=1e-5):
            out["_note"] = "Split softmax output does not match standard softmax within tolerance"
            return out

        out["softmax_match"] = 1.0
    except Exception as e:
        out["_note"] = f"Error during split softmax execution: {type(e).__name__}: {str(e)[:120]}"
    return out
