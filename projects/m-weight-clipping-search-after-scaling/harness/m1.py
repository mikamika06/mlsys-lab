import numpy as np
import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from awq_clip.quant import quantize_and_reconstruct

    out = {"mse_match": 0.0}

    w_reshaped = ref.W_TEST.reshape(-1, ref.GROUP_SIZE)
    w_max = np.max(np.abs(w_reshaped), axis=1, keepdims=True) * 0.75

    try:
        want = ref.quantize_and_reconstruct_ref(w_reshaped, w_max, ref.N_BITS)
        got = quantize_and_reconstruct(w_reshaped, w_max, ref.N_BITS)

        if np.allclose(got, want, atol=1e-5):
            out["mse_match"] = 1.0
        else:
            out["_note"] = "Reconstructed weights do not match reference implementation."
    except Exception as e:
        out["_note"] = f"Failed to execute quantize_and_reconstruct: {e}"

    return out
