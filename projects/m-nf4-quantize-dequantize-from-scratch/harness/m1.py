import ref
import numpy as np


def check(workdir):
    out = {"codebook_match": 0.0}
    try:
        from quant.codebook import get_nf4_codebook
        got = get_nf4_codebook()
        want, _ = ref.compute_reference_codebooks()
        if got is not None and np.allclose(got, want, atol=1e-5):
            out["codebook_match"] = 1.0
        else:
            out["_note"] = f"Codebook does not match reference. Got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"Error importing or executing get_nf4_codebook: {e}"
    return out
