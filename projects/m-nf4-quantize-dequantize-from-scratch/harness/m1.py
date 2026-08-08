import ref
import numpy as np


def check(workdir):
    out = {"codebook_match": 0.0}
    want = ref.build_nf4_codebook()

    try:
        from nf4.codebooks import build_nf4_codebook
        got = build_nf4_codebook()
        if np.allclose(want, got, atol=1e-5):
            out["codebook_match"] = 1.0
        else:
            out["_note"] = f"Codebook mismatch. Got {got[:3]}..., expected {want[:3]}..."
    except NotImplementedError:
        pass
    except Exception as e:
        out["_note"] = str(e)

    return out
